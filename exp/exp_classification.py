from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_count = 0
        preds = []
        trues = []
        self.model.eval()
        max_val_steps = getattr(self.args, 'max_val_steps', -1)
        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                if max_val_steps > 0 and i >= max_val_steps:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                padding_mask = padding_mask.float().to(self.device, non_blocking=non_blocking)
                label = label.to(self.device, non_blocking=non_blocking)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                loss_sum += loss.detach().float()
                loss_count += 1

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = (loss_sum / max(1, loss_count)).item()

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

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
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                if max_train_steps > 0 and i >= max_train_steps:
                    break
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                padding_mask = padding_mask.float().to(self.device, non_blocking=non_blocking)
                label = label.to(self.device, non_blocking=non_blocking)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().squeeze(-1))
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
                    scaler.unscale_(model_optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = (train_loss_sum / max(1, train_loss_count)).item()
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        max_test_steps = getattr(self.args, 'max_test_steps', -1)
        use_amp = bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")
        non_blocking = (self.device.type == "cuda")
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                if max_test_steps > 0 and i >= max_test_steps:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=non_blocking)
                padding_mask = padding_mask.float().to(self.device, non_blocking=non_blocking)
                label = label.to(self.device, non_blocking=non_blocking)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
