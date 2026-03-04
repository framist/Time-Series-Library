from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

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

    def vali(self, vali_data, vali_loader, criterion):
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_count = 0
        self.model.eval()
        max_val_steps = getattr(self.args, 'max_val_steps', -1)
        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                if max_val_steps > 0 and i >= max_val_steps:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
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
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if max_train_steps > 0 and i >= max_train_steps:
                    break
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
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
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        max_test_steps = getattr(self.args, 'max_test_steps', -1)
        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if max_test_steps > 0 and i >= max_test_steps:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                # reconstruction
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x.float(), outputs.float()), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if max_test_steps > 0 and i >= max_test_steps:
                break
            batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
            # reconstruction
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x.float(), outputs.float()), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return
