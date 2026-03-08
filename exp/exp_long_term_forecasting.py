from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from layers.HSPMF import crps_from_pmf, nll_loss_from_pmf, targets_to_grid_index
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import json
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _core_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _split_model_output(self, model_output):
        if isinstance(model_output, dict):
            pred = model_output.get("pred")
            if pred is None:
                raise ValueError("HSPMF 输出缺少 pred 字段")
            return pred, model_output
        return model_output, None

    def _slice_forecast_tensors(self, pred, batch_y, aux=None):
        f_dim = -1 if self.args.features == 'MS' else 0
        pred = pred[:, -self.args.pred_len:, f_dim:]
        true = batch_y[:, -self.args.pred_len:, f_dim:]
        posterior = None
        if aux is not None and "posterior" in aux:
            posterior = aux["posterior"][:, -self.args.pred_len:, f_dim:, :]
        return pred, true, posterior

    def _compute_loss(self, pred, true, posterior, criterion):
        hspmf_loss = str(getattr(self.args, "hspmf_loss", "mse")).lower()
        if hspmf_loss == "nll":
            if posterior is None:
                raise RuntimeError("hspmf_loss=nll 需要模型返回 posterior")
            decoder = getattr(self._core_model(), "hspmf_decoder", None)
            if decoder is None:
                raise RuntimeError("当前模型未挂载 hspmf_decoder")
            target_idx = targets_to_grid_index(true, decoder.x_grid)
            return nll_loss_from_pmf(posterior, target_idx)
        return criterion(pred, true)

    def _compute_hspmf_eval_metrics(self, posterior, true):
        decoder = getattr(self._core_model(), "hspmf_decoder", None)
        if decoder is None:
            raise RuntimeError("当前模型未挂载 hspmf_decoder")
        target_idx = targets_to_grid_index(true, decoder.x_grid)
        nll = nll_loss_from_pmf(posterior, target_idx)
        crps = crps_from_pmf(posterior, true, decoder.x_grid)
        return nll, crps, int(true.numel())

    def _current_hspmf_beta(self):
        core_model = self._core_model()
        if not hasattr(core_model, "get_hspmf_beta"):
            return None
        beta = core_model.get_hspmf_beta()
        if beta is None:
            return None
        return float(beta.detach().cpu().item())

    def vali(self, vali_data, vali_loader, criterion):
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_count = 0
        self.model.eval()
        max_val_steps = getattr(self.args, 'max_val_steps', -1)
        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                if max_val_steps > 0 and i >= max_val_steps:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                batch_y = batch_y.float().to(self.device, non_blocking=non_blocking)

                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=non_blocking)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # encoder - decoder
                with torch.cuda.amp.autocast(enabled=use_amp):
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, aux = self._split_model_output(model_output)
                outputs, batch_y, posterior = self._slice_forecast_tensors(outputs, batch_y, aux)
                loss = self._compute_loss(outputs, batch_y, posterior, criterion)
                loss_sum += loss.detach().float()
                loss_count += 1
        total_loss = (loss_sum / max(1, loss_count)).item()
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            train_loss_count = 0

            self.model.train()
            epoch_time = time.time()
            max_train_steps = getattr(self.args, 'max_train_steps', -1)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if max_train_steps > 0 and i >= max_train_steps:
                    break
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                batch_y = batch_y.float().to(self.device, non_blocking=non_blocking)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=non_blocking)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if use_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, aux = self._split_model_output(model_output)
                        outputs, batch_y_sliced, posterior = self._slice_forecast_tensors(outputs, batch_y, aux)
                        loss = self._compute_loss(outputs, batch_y_sliced, posterior, criterion)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, aux = self._split_model_output(model_output)
                    outputs, batch_y_sliced, posterior = self._slice_forecast_tensors(outputs, batch_y, aux)
                    loss = self._compute_loss(outputs, batch_y_sliced, posterior, criterion)
                train_loss_sum += loss.detach().float()
                train_loss_count += 1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = (train_loss_sum / max(1, train_loss_count)).item()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # ── 外推评估：计算训练集每通道 max(|x|) ──
        extrap_eval = getattr(self.args, 'wv_extrap_eval', False)
        train_max_abs = None
        if extrap_eval:
            train_data, _ = self._get_data(flag='train')
            # 获取原始值空间下的训练集最大绝对值
            # 注意：对于 standard/prior(offset!=0) 等含平移项的逆变换，
            # 必须先 inverse_transform 再取 abs，二者不可交换。
            raw_train = train_data.data_x  # [N, M] 或 [N, T, M]
            if raw_train.ndim == 2:
                train_flat = raw_train
            else:
                train_flat = raw_train.reshape(-1, raw_train.shape[-1])
            if train_data.scale:
                train_orig = train_data.inverse_transform(train_flat)
                train_max_abs = np.abs(train_orig).max(axis=0)  # [M]
            else:
                train_max_abs = np.abs(train_flat).max(axis=0)  # [M]
            print(f'[ExtrapEval] train_max_abs per channel: {train_max_abs}')
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        max_test_steps = getattr(self.args, 'max_test_steps', -1)
        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")
        hspmf_nll_sum = 0.0
        hspmf_crps_sum = 0.0
        hspmf_metric_count = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if max_test_steps > 0 and i >= max_test_steps:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                batch_y = batch_y.float().to(self.device, non_blocking=non_blocking)

                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=non_blocking)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # encoder - decoder
                with torch.cuda.amp.autocast(enabled=use_amp):
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs, aux = self._split_model_output(model_output)
                outputs_full = outputs[:, -self.args.pred_len:, :]
                batch_y_full = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs_sliced, batch_y_sliced, posterior = self._slice_forecast_tensors(outputs, batch_y, aux)
                if posterior is not None:
                    batch_nll, batch_crps, batch_count = self._compute_hspmf_eval_metrics(posterior, batch_y_sliced)
                    hspmf_nll_sum += float(batch_nll.detach().cpu().item()) * batch_count
                    hspmf_crps_sum += float(batch_crps.detach().cpu().item()) * batch_count
                    hspmf_metric_count += batch_count

                outputs = outputs_full.detach().cpu().numpy()
                batch_y = batch_y_full.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))

        # ── 外推评估：域内/域外分组统计 ──
        extrap_info = ''
        if extrap_eval and train_max_abs is not None and self.args.inverse:
            f_dim = -1 if self.args.features == 'MS' else 0
            # train_max_abs 可能是全通道，需要按 f_dim 切片
            if f_dim == -1:
                t_max = train_max_abs[-1:]  # [1]
            else:
                t_max = train_max_abs  # [M]
            # 每个样本：检查 ground truth 是否超出训练集范围
            # trues: [N, T, M']
            sample_max = np.abs(trues).max(axis=1)  # [N, M']
            ood_mask = np.any(sample_max > t_max, axis=1)  # [N]
            n_in = int((~ood_mask).sum())
            n_out = int(ood_mask.sum())
            print(f'[ExtrapEval] in-domain: {n_in}, out-of-domain: {n_out} (total: {len(ood_mask)})')
            if n_in > 0:
                mae_in, mse_in, _, _, _ = metric(preds[~ood_mask], trues[~ood_mask])
                print(f'[ExtrapEval] IN-domain  mse:{mse_in:.6f}, mae:{mae_in:.6f}')
            else:
                mse_in = mae_in = float('nan')
                print('[ExtrapEval] no in-domain samples')
            if n_out > 0:
                mae_out, mse_out, _, _, _ = metric(preds[ood_mask], trues[ood_mask])
                print(f'[ExtrapEval] OUT-domain mse:{mse_out:.6f}, mae:{mae_out:.6f}')
            else:
                mse_out = mae_out = float('nan')
                print('[ExtrapEval] no out-of-domain samples')
            extrap_info = f', mse_in:{mse_in}, mae_in:{mae_in}, mse_out:{mse_out}, mae_out:{mae_out}, n_in:{n_in}, n_out:{n_out}'

        hspmf_info = ''
        if hspmf_metric_count > 0:
            hspmf_nll = hspmf_nll_sum / hspmf_metric_count
            hspmf_crps = hspmf_crps_sum / hspmf_metric_count
            hspmf_beta = self._current_hspmf_beta()
            hspmf_info = f', hspmf_nll:{hspmf_nll}, hspmf_crps:{hspmf_crps}'
            if hspmf_beta is not None:
                hspmf_info += f', hspmf_beta:{hspmf_beta}'
            print(f'[HSPMF] nll:{hspmf_nll:.6f}, crps:{hspmf_crps:.6f}, beta:{hspmf_beta}')

            decoder = getattr(self._core_model(), "hspmf_decoder", None)
            hspmf_metrics = {
                "nll": hspmf_nll,
                "crps": hspmf_crps,
                "beta": hspmf_beta,
                "space": "decoder_grid",
            }
            if decoder is not None:
                hspmf_metrics.update(
                    {
                        "x_grid_min": float(decoder.x_grid[0].detach().cpu().item()),
                        "x_grid_max": float(decoder.x_grid[-1].detach().cpu().item()),
                        "grid_size": int(decoder.x_grid.numel()),
                    }
                )
            with open(os.path.join(folder_path, 'hspmf_dist_metrics.json'), 'w', encoding='utf-8') as fh:
                json.dump(hspmf_metrics, fh, ensure_ascii=False, indent=2)

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw) + extrap_info + hspmf_info)
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
