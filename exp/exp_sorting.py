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

warnings.filterwarnings("ignore")


class Exp_Sorting(Exp_Basic):
    """解交织任务（也可以称作 token classification）"""

    def __init__(self, args):
        super(Exp_Sorting, self).__init__(args)
        assert args.data == "EBDSC_2nd", "only support EBDSC_2nd dataset"

    def _build_model(self):
        # 所需参数：
        # self.args.seq_len = self.args.pred_len 窗口长度
        # self.args.enc_in 通道数
        # self.args.c_out 分类数
        print(f"Exp_Sorting needed args: {self.args.seq_len=}, {self.args.enc_in=}, {self.args.c_out=}")
        self.args.num_class = self.args.c_out
        self.args.pred_len = 0
        if self.args.embed == "prepos":
            print(f'prepos args: {self.args.d_model=}, {self.args.wve_mask=}, {self.args.wve_mask_hard=}')
        

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
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
        def criterion(outputs: torch.FloatTensor, targets: torch.Tensor):
            # loss_1 = focal_loss(outputs.view(-1, 12), targets.view(-1), alpha=None, reduction='mean')
            loss_1 = nn.CrossEntropyLoss()(outputs.view(-1, 12), targets.view(-1))
            return loss_1

        return criterion

    def vali(self, _, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, None, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=-1)  # (B, L, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=-1).flatten().cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        """
        TODO: lr schedule
        """
        recorder = ExperimentRecorder(setting)

        _, train_loader = self._get_data(flag="TRAIN")
        _, vali_loader = self._get_data(flag="VALID")
        _, test_loader = self._get_data(flag="TEST_MINI")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            # 重生成数据集
            if epoch % self.args.data_regen_epoch == 0 and epoch >= 0:
                _, train_loader = self._get_data(flag="TRAIN")
                _, vali_loader = self._get_data(flag="VALID")
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x: torch.Tensor = batch_x.float().to(self.device)
                label_y: torch.Tensor = label_y.to(self.device)

                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, label_y.long())
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(_, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(_, test_loader, criterion)
            left_time = (time.time() - time_now) / (epoch + 1) * (self.args.train_epochs - epoch - 1)

            print(f"E{epoch + 1} | TrL: {train_loss:.3f} VaL: {vali_loss:.3f} VaA: {val_accuracy:.3f} TeL: {test_loss:.3f} TeA: {test_accuracy:.3f}, left time: {left_time/3600:.3f}h")
            
            recorder.add_record("train", train_loss)
            recorder.add_record("valid", vali_loss)
            recorder.add_record("test", test_loss)
            recorder.add_record("acc", test_accuracy)
            recorder.plot_loss(self.args.des, save_fig=True)

            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        recorder.save_records()
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        Test the model in three scenarios

        Args:
            setting (str): The setting name used for loading the model and saving results.
            test (int, optional): If 1, only perform testing and load the model. If 0, use the already loaded best model. Defaults to 0.
                test=1 时 只进行测试，所以要先加载模型；而 test=0 时已加载最优模型

        Returns:
            None
        """
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        fs = ["SCENE_I", "SCENE_II", "SCENE_III", "TEST_ALL"]
        file_name = "result_sorting.txt"
        folder_path = f"./results/{setting}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, file_name), "a") as f:
            f.write(f"{setting}\n")
            for flag in fs:
                _, test_loader = self._get_data(flag)

                preds = []
                trues = []

                self.model.eval()
                with torch.no_grad():
                    for i, (batch_x, label) in enumerate(test_loader):
                        batch_x = batch_x.float().to(self.device)
                        label = label.to(self.device)

                        outputs = self.model(batch_x, None, None, None)

                        preds.append(outputs.detach())
                        trues.append(label)

                preds = torch.cat(preds, 0)
                trues = torch.cat(trues, 0)
                # print("test shape:", preds.shape, trues.shape)

                probs = torch.nn.functional.softmax(preds)  # (B, L, num_classes) est. prob. for each class and sample
                predictions = (
                    torch.argmax(probs, dim=-1).flatten().cpu().numpy()
                )  # (total_samples,) int class index for each sample
                trues = trues.flatten().cpu().numpy()
                log = confusion_matrix(predictions, trues, folder_path, plot_name=setting, tag_len=self.args.c_out)
                
                print(f"{flag:<8} {log}")
                f.write(f"{flag:<8} {log}\n")
                
            f.write("\n")
            f.write("\n")

        return


import json
import matplotlib.pyplot as plt
import datetime


class ExperimentRecorder:
    """实验数据记录器"""

    def __init__(self, exp_name, save_dir="./results"):
        self.exp_name = exp_name
        self.save_dir = os.path.join(save_dir, exp_name)
        self.loss_record: dict[str, list[float]] = {"train": [], "valid": [], "test": [], "acc": []}  # test acc

        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def add_record(self, record_type, value):
        """添加记录
        Args:
            record_type: 记录类型 ('train'/'valid'/'test'/'acc')
            value: 要记录的值
        """
        if record_type in self.loss_record:
            self.loss_record[record_type].append(float(value))
        else:
            raise ValueError(f"Unknown record type: {record_type}")

    def save_records(self):
        """保存记录到 JSON 文件"""
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(self.save_dir, f"experiment_records_{now}.json")
        with open(save_path, "w") as f:
            json.dump(self.loss_record, f, indent=2)

    def load_records(self):
        """从 JSON 文件加载记录"""
        load_path = os.path.join(self.save_dir, "experiment_records.json")
        if os.path.exists(load_path):
            with open(load_path, "r") as f:
                self.loss_record = json.load(f)

    def plot_loss(self, plot_name: str, save_fig=True):
        loss_record = self.loss_record
        plt.figure()
        plt.plot(loss_record["train"], label="train", linestyle="-", marker=".", linewidth=1, alpha=0.6)
        plt.plot(loss_record["valid"], label="valid", linestyle="-", marker=".", linewidth=1, alpha=0.6)
        plt.plot(loss_record["test"], label="test", alpha=0.9)
        plt.plot(loss_record["acc"], label="acc", alpha=0.9)
        plt.grid()
        plt.ylim(0, 3)
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f'{plot_name}\nmax acc: {max(loss_record["acc"]):.3f} min loss:{min(loss_record["test"]):.5f}')
        if save_fig:
            plt.savefig(f"{self.save_dir}/{plot_name}_loss.pdf")
        plt.show()
        plt.close()


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def confusion_matrix(
    predictions, targets, save_dir: str, plot_name: str = None, tag_len: int = 12, average: str = "weighted"
):
    """
    混淆矩阵
    """
    confusion_matrix = np.zeros((tag_len, tag_len))

    for target, prediction in zip(targets, predictions):
        confusion_matrix[target, prediction] += 1

    # 计算精确率、召回率、F1
    # 注：average='macro' 意味着每个类别的权重相同
    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    f1 = f1_score(targets, predictions, average=average)
    log = f"Acc: {acc*100:.3f}, Pre: {precision*100:.3f}, Rec: {recall*100:.3f}, F1: {f1*100:.3f} [{average}]"
    print(log)

    if plot_name:
        plt.figure()
        plt.imshow(confusion_matrix / np.maximum(1, np.sum(confusion_matrix, axis=1)[:, None]))
        # 同时在方格内显示数值
        for i in range(tag_len):
            for j in range(tag_len):
                plt.text(j, i, f"{confusion_matrix[i, j]:.0f}", ha="center", va="center", color="blue")
        plt.title("confusion matrix")
        plt.xlabel("prediction")
        plt.ylabel("target")
        plt.colorbar()
        plt.title(f"{plot_name}\n{log}")
        plt.savefig(f"{save_dir}/CM_{plot_name}.pdf")
        plt.show()

    return log
