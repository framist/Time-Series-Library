import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_EEG(Dataset):
    """脑电信号的自定义数据集
    弃用，因：`Dataset_Custom` 模板非分类任务"""

    def __init__(
        self,
        args,
        root_path,  # 父文件夹，使用此需指定 `--root_path ./dataset/EEG/`
        flag="train",
        size=None,  # [seq_len, label_len, pred_len]
        features="S",
        data_path="data/", # 预定好的数据集文件夹
        # target="OT",
        scale=True, # 是否标准，（在此设定，非传参设定）
        # timeenc=0,
        # freq="h",
        # seasonal_patterns=None,
    ):
        """e.g. 
        ```
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        ```"""
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        # self.target = target
        self.scale = scale
        # self.timeenc = timeenc
        # self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
    def __read_data__(self):
        """这里他本身不应该加双下划线的"""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


# -------------------------------------------------- EEG --------------------------------------------------


import scipy
import scipy.signal
from scipy import signal
from scipy.stats import kurtosis, skew
from torch.utils.data import ConcatDataset, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


class _MyEEGDataset(Dataset):
    def __init__(
        self, data: torch.FloatTensor, labels: torch.LongTensor, extract_feats: bool = False, extra_feats=None
    ):
        """初始化数据集

        Args:
            data (torch.FloatTensor): _description_
            labels (torch.LongTensor): _description_
            extract_feats (bool, optional): 是否手动提取特征 (N*9) 。Defaults to False.
            extra_feats (_type_, optional): 外部输入的额外特征。Defaults to None.
        """
        self.data = data
        self.labels = labels
        self.extract_feats = extract_feats
        self.extra_feats = extra_feats
        if self.extra_feats is not None:
            self.extra_feats = torch.tensor(self.extra_feats, dtype=torch.float32)
            # self.extra_feats = torch.zeros_like(self.extra_feats, dtype=torch.float32)
            assert len(self.extra_feats) == len(self.data), f"{len(self.extra_feats)} != {len(self.data)}"

        if self.extract_feats:
            x = self.data.numpy()
            mmean = np.mean(x, axis=2)
            sstd = np.std(x, axis=2, ddof=1)
            kkur = kurtosis(x, axis=2, fisher=False)
            sskew = skew(x, axis=2)

            # 计算相对功率特征
            fre, psd = signal.welch(x, fs=250, window="hann", axis=2)
            power_all = np.sum(psd[:, :, (fre > 0.5) & (fre <= 45)], axis=2) + 0.000001
            power_delta = np.sum(psd[:, :, (fre > 0.5) & (fre <= 4)], axis=2) / power_all
            power_theta = np.sum(psd[:, :, (fre > 4) & (fre <= 8)], axis=2) / power_all
            power_alpha = np.sum(psd[:, :, (fre > 8) & (fre <= 13)], axis=2) / power_all
            power_beta = np.sum(psd[:, :, (fre > 13) & (fre <= 30)], axis=2) / power_all
            power_gamma = np.sum(psd[:, :, (fre > 30) & (fre <= 45)], axis=2) / power_all

            # 特征拼接
            features = np.concatenate(
                (
                    mmean,
                    sstd,
                    kkur,
                    sskew,
                    power_delta,
                    power_theta,
                    power_alpha,
                    power_beta,
                    power_gamma,
                ),
                axis=1,
            )

            # 将异常值替换为 0
            features = np.nan_to_num(features)

            # 特征归一化
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
            if extra_feats is None:
                self.data = torch.tensor(features, dtype=torch.float32)
            else:
                self.extra_feats = torch.cat((self.extra_feats, torch.tensor(features, dtype=torch.float32)), dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.extra_feats is not None:
            return self.data[idx].unsqueeze(0), self.extra_feats[idx], self.labels[idx]
        if self.extract_feats:
            sample = self.data[idx]
        else:
            sample = self.data[idx].permute(1, 0)  # [T,C]
        label = self.labels[idx]
        return sample, label


class EEGloader(Dataset):
    """
    ```python
    scenes = ["无电磁环境", "全任务模拟器", "演示验证系统"]
    tasks = ["脑负荷", "脑疲劳", "脑警觉", "注意力"]
    [
            f"data/{scenes[0]}_{task}-1_data.mat",
            f"data/{scenes[0]}_{task}-2_data.mat",
    ],
    ```
    """

    def __init__(
        self,
        args,
        root_path,  # 父文件夹，使用此需指定 `--root_path ./dataset/EEG/`
        flag=None,
    ):
        """
        e.g.
        ```
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        ```
        """
        self.args = args
        self.root_path = root_path
        # assert flag in ["train", "test", "val"]
        self.flag = flag

        scenes = ["无电磁环境", "全任务模拟器", "演示验证系统"]
        tasks = ["脑负荷", "脑疲劳", "脑警觉", "注意力"]

        # - 训练
        task = tasks[0]
        scene = scenes[0]
        self._processor(
            file_path_0=os.path.join(root_path, f"data/{scene}_{task}-1_data.mat"),
            file_path_1=os.path.join(root_path, f"data/{scene}_{task}-2_data.mat"),
            if_norm=True,
        )

        self._create_split_dataloaders()

        # # - 真正的 test
        # task = tasks[0]
        # scene = scenes[2]
        # self._processor(
        #     file_path_0=os.path.join(root_path, f"data/{scene}_{task}-1_data.mat"),
        #     file_path_1=os.path.join(root_path, f"data/{scene}_{task}-2_data.mat"),
        #     if_norm=True,
        # )
        # self._create_test_dataset()

    def _processor(
        self, file_path_0: str, file_path_1: str, if_norm: bool = False, if_downsample: bool = False, **kwargs
    ):
        """Load data from two files and concatenate them."""
        TIME_POINTS = 1000
        print(f"Loading data {file_path_0} & {file_path_1}")
        self.data_lable_0: np.ndarray = scipy.io.loadmat(file_path_0)["Data"]
        self.data_lable_1: np.ndarray = scipy.io.loadmat(file_path_1)["Data"]

        assert self.data_lable_0.shape[1] % TIME_POINTS == 0
        assert self.data_lable_1.shape[1] % TIME_POINTS == 0

        print(f"{self.data_lable_0.shape = } {self.data_lable_1.shape = }")

        data = np.concatenate((self.data_lable_0, self.data_lable_1), axis=1)  # (8, TIME_POINTS * N)
        # 每个通道 Normalization
        # data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        data = data.reshape((8, -1, TIME_POINTS)).transpose((1, 0, 2))  # (N, 8, TIME_POINTS)

        labels = np.concatenate(
            (
                np.zeros(self.data_lable_0.shape[1] // TIME_POINTS),
                np.ones(self.data_lable_1.shape[1] // TIME_POINTS),
            )
        )
        data = torch.tensor(data, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)
        assert data.shape == (len(self.labels), 8, TIME_POINTS)

        if if_downsample:
            # * 降采样
            # data = self.data[:, :, ::2]
            data = torch.tensor(
                scipy.signal.resample(self.data.numpy(), TIME_POINTS // 2, axis=-1),
                dtype=torch.float32,
            )
            TIME_POINTS = TIME_POINTS // 2

        self.max_seq_len = TIME_POINTS
        if if_norm:
            # * 最后一维 Normalization
            data = (data - data.mean(dim=-1, keepdim=True)) / data.std(dim=-1, keepdim=True)

        self.data = data
        # self.feature_df = data
        self.dataset = _MyEEGDataset(data, self.labels, **kwargs)
        self.class_names = ["0", "1"]
        self.enc_in = data.shape[1]

    def _create_split_dataloaders(self, train_ratio: float = 0.9) -> tuple[DataLoader, DataLoader]:
        """创建训练集和验证集的 DataLoader"""
        train_size = int(train_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        # self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        # 固定划分 因为存在多次调用建立，所以划分需要固定
        self.train_dataset = Subset(self.dataset, range(0, train_size))
        self.val_dataset = Subset(self.dataset, range(train_size, train_size + val_size))

        print(f"EEG 训练集大小：{len(self.train_dataset)} 验证集大小：{len(self.val_dataset)}")

    def _create_test_dataset(self) -> tuple[DataLoader, DataLoader]:
        """创建训练集和验证集的 DataLoader"""
        self.train_dataset = None
        self.test_dataset = self.dataset

    def __getitem__(self, ind):
        if self.flag == "TRAIN":
            return self.train_dataset[ind]
        elif self.flag == "TEST":
            return self.val_dataset[ind]
            # return self.test_dataset[ind]

    def __len__(self):
        if self.flag == "TRAIN":
            return len(self.train_dataset)
        elif self.flag == "TEST":
            return len(self.val_dataset)
            # return len(self.test_dataset)


class EEGloaderMix(Dataset):
    """混合场景"""
    def __init__(
        self,
        args,
        root_path,  # 父文件夹，使用此需指定 `--root_path ./dataset/EEG/`
        flag=None,
    ):
        """
        e.g.
        ```
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        ```
        """
        self.args = args
        self.root_path = root_path
        # assert flag in ["train", "test", "val"]
        self.flag = flag

        scenes = ["无电磁环境", "全任务模拟器", "演示验证系统"]
        tasks = ["脑负荷", "脑疲劳", "脑警觉", "注意力"]

        # - 训练
        train_datas = []
        for scene in scenes:
            for task in [tasks[1]]:
                train_datas.append(
                    self._processor(
                        file_path_0=os.path.join(root_path, f"data/{scene}_{task}-1_data.mat"),
                        file_path_1=os.path.join(root_path, f"data/{scene}_{task}-2_data.mat"),
                    )
                )

        # 合并数据集
        train_data = ConcatDataset(train_datas)
        print(f"合并数据集大小：{len(train_data)}")
        self.dataset = train_data

        self.max_seq_len = 1000
        self.class_names = ["0", "1"]
        self.enc_in = 8
        self._create_split_dataloaders()

    @staticmethod
    def _processor(file_path_0: str, file_path_1: str, if_norm: bool = True, if_downsample: bool = False, **kwargs):
        """Load data from two files and concatenate them."""
        TIME_POINTS_O = 1000
        print(f"Loading data {file_path_0} & {file_path_1}")
        data_lable_0: np.ndarray = scipy.io.loadmat(file_path_0)["Data"]
        data_lable_1: np.ndarray = scipy.io.loadmat(file_path_1)["Data"]

        assert data_lable_0.shape[1] % TIME_POINTS_O == 0
        assert data_lable_1.shape[1] % TIME_POINTS_O == 0

        print(f"{data_lable_0.shape = } {data_lable_1.shape = }")

        data = np.concatenate((data_lable_0, data_lable_1), axis=1)  # (8, TIME_POINTS * N)
        # TODO Normalization
        if if_norm:
            # * Normalization
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        # data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        data = data.reshape((8, -1, TIME_POINTS_O)).transpose((1, 0, 2))  # (N, 8, TIME_POINTS)

        labels = np.concatenate(
            (
                np.zeros(data_lable_0.shape[1] // TIME_POINTS_O),
                np.ones(data_lable_1.shape[1] // TIME_POINTS_O),
            )
        )
        data = torch.tensor(data, dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.long)
        assert data.shape == (len(labels), 8, TIME_POINTS_O)

        if if_downsample:
            # * 降采样
            # data = data[:, :, ::2]
            data = torch.tensor(
                scipy.signal.resample(data.numpy(), TIME_POINTS_O // 2, axis=-1),
                dtype=torch.float32,
            )
            TIME_POINTS_O = TIME_POINTS_O // 2

        # self.data = data
        return _MyEEGDataset(data, labels, **kwargs)

    def _create_split_dataloaders(self, fold_index: int = 2, n_splits: int = 5) -> tuple[DataLoader, DataLoader]:
        """创建训练集和验证集的 DataLoader，使用 K 折交叉验证"""
        indices = list(range(len(self.dataset)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(indices))

        # 检查 fold_index 是否有效
        if fold_index < 0 or fold_index >= n_splits:
            raise ValueError(f"fold_index 应该在 0 和 {n_splits - 1} 之间")

        train_indices, val_indices = splits[fold_index]

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        print(f"EEG 训练集大小：{len(self.train_dataset)} 验证集大小：{len(self.val_dataset)}")

    def _create_test_dataset(self) -> tuple[DataLoader, DataLoader]:
        """创建训练集和验证集的 DataLoader"""
        self.train_dataset = None
        self.test_dataset = self.dataset

    def __getitem__(self, ind):
        # TODO
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            # 获取原始数据和标签
            batch_x, labels = self.train_dataset[ind]  # batch_x: (8, TIME_POINTS_O), labels: scalar

            # 添加批次维度以适配增广函数
            batch_x = batch_x.unsqueeze(0)  # (1, 8, TIME_POINTS_O)
            labels = labels.unsqueeze(0)    # (1,)

            # 执行数据增广
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)
            # 假设 run_augmentation_single 返回增广后的 batch_x 和 labels

            # 移除批次维度
            batch_x = batch_x.squeeze(0)  # (8, TIME_POINTS_O)
            labels = labels.squeeze(0)    # scalar

            # 归一化处理
            # batch_x = self.instance_norm(batch_x)

            return batch_x, labels

        if self.flag == "TRAIN":
            return self.train_dataset[ind]
        elif self.flag == "TEST":
            return self.val_dataset[ind]

    def __len__(self):
        if self.flag == "TRAIN":
            return len(self.train_dataset)
        elif self.flag == "TEST":
            return len(self.val_dataset)


# -------------------------------------------------- Radio --------------------------------------------------
from data_provider.ebdsc_2nd import mix_data_gen, to_dict, read_dfs, target_domain_data_gen


class EBDSC_2nd(Dataset):
    TAG_LEN = 12
    # WINDOW_SIZE = 1024
    DROP_SIG_RATIO = 0.99
    
    def __init__(
        self,
        args,
        root_path,
        win_size,
        flag=None,
        if_emb=False,
    ):
        """

        Args:
            args (_type_): _description_
            root_path 父文件夹，使用此需指定 `--root_path ./dataset/EBDSC-2nd/` 暂时弃用
            flag (_type_, optional): _description_. Defaults to None.
        """
        assert args.enc_in == 5
        assert args.c_out == self.TAG_LEN
        assert flag in ["TRAIN", "VALID", "TEST"]
        
        
        self.args = args
        self.root_path = root_path
        self.flag = flag

        df_list, test_df_list = read_dfs()

        if flag == "TRAIN":
            d_train = mix_data_gen(df_list, 100, 100, 25, True)
            self.inputs, self.targets = self.make_data(d_train)
        elif flag == "VALID":
            d_valid = mix_data_gen(df_list, 20, 50, 20, True)
            self.inputs, self.targets = self.make_data(d_valid)
        elif flag == "TEST":        # TODO
            self.inputs, self.targets = self.make_data(target_domain_data_gen(test_df_list[2], 20, 50))
        elif flag == "TEST_ALL":    # TODO
            d_test = []
            for test_df in test_df_list:
                for i in range(0, test_df.shape[0] - win_size, win_size):
                    df_window = test_df.iloc[i : i + win_size, :]
                    m2, m3 = to_dict(df_window)
                    d_test.append([m2, m3])
            self.inputs, self.targets = self.make_data(d_test)
            
        if not if_emb:
            return
        else:
            print(f'needed args: {args.d_model=}, {args.wve_mask=}, {args.wve_mask_hard=}')
            self.d_model = args.d_model
            self.hard = args.wve_mask_hard if flag == "TRAIN" else 0
            
            self.d_step = 8
            # mod_max = 65536 = 2**N 因为满足 2 ** (N*d_step/d_model) == 2
            mod_max = 65536
            # div_term
            # self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device)
            # self.div_term = (1. / (65536 ** (torch.arange(0, d_model, 2) / d_model))).to(device)
            self.div_term = 1.0 / (mod_max ** (torch.arange(0, self.d_model, self.d_step) / self.d_model))
            # mod_d = lambda d: mod_max ** (np.floor(d/d_step)*d_step / d_model)
            # 根据 mod 查维度，上界 <
            self.d_mod = lambda m: np.floor(self.d_step * np.log2(m)).astype(np.int64) + 1

            if args.wve_mask == 'r':
                self.f_mask  = lambda x: torch.rand_like(x) * 2 - 1
            elif args.wve_mask == 'm':
                self.f_mask  = lambda x: torch.mean(x, axis=0)
            elif args.wve_mask == 'c':
                self.f_mask  = lambda x: torch.zeros_like(x)
            else:
                raise ValueError(f'{args.wve_mask=} not supported')

    @staticmethod
    def make_data(d: list[list[np.ndarray]]):
        """生成数据集
        Args:
            d 数据集
        Returns:
            inputs2 = 特征
            targets = [[TAG] * 时间窗长度] * 样本数 one-hot 编码
        """
        inputs = np.array([i[0] for i in d], dtype=np.float32)
        targets = np.array([i[-1] for i in d], dtype=np.int64) - 1
        return torch.FloatTensor(inputs), torch.LongTensor(targets)

    def __getitem__(self, ind):
        idx = ind

        if not hasattr(self, "d_model"):
            inputs = self.inputs[idx]
            # 除第一个维度，每一个维度正则化
            inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
            inputs[:, 0] = (self.inputs[idx][:, 0] / 5e5 - 0.0002) / 0.0005
            # inputs = self.inputs[idx] / 65536 # old use
            # inputs = self.inputs[idx]
            # # !!! input[:, 0] = label
            # inputs[:, 0] = self.targets[idx][:, 0]
            return inputs, self.targets[idx]

        # * POS [1024, 5] -> [1024, 5, 128]
        positions: torch.FloatTensor = self.inputs[idx]  # [1024, 5]
        win_size, input_channels = positions.size()
        pe = torch.zeros(win_size, input_channels, self.d_model)

        positions = positions.unsqueeze(-1)  # [1024, 5, 1]

        # pe[:, :, 0::2] = torch.sin(positions * self.div_term * torch.pi)
        # pe[:, :, 1::2] = torch.cos(positions * self.div_term * torch.pi)

        for i in range(self.d_step):
            pe[:, :, i :: self.d_step] = (positions * self.div_term + 1 / self.d_step * i) % 1 * 2 - 1
            # # linearV
            # pe[:, :, i::self.d_step] = torch.absolute((positions * self.div_term + 1 / self.d_step * i) % 1 - 0.5) * 2 - 1

        if self.hard:

            if np.random.rand() < 1 * self.hard:
                # 1 RF mimax in 5 ~ 28
                mask_d_min = np.random.randint(self.d_mod(5), self.d_mod(60))
                pe[:, 1, mask_d_min:] = self.f_mask(pe[:, 1, mask_d_min:])

            if np.random.rand() < 1 * self.hard:
                # 2 PW * 10 mimax in 6 ~ 50
                mask_d_min = np.random.randint(self.d_mod(6), self.d_mod(100))
                pe[:, 2, mask_d_min:] = self.f_mask(pe[:, 2, mask_d_min:])

            if np.random.rand() < 0.1 * self.hard:
                # 3 RF ?
                pe[:, 3, :] = self.f_mask(pe[:, 3, :])

            if np.random.rand() < 0.5 * self.hard:
                # 4 DOA mimax in 6 ~ 7
                mask_d_min = np.random.randint(self.d_mod(6), self.d_mod(14))
                pe[:, 4, mask_d_min:] = self.f_mask(pe[:, 4, mask_d_min:])

        return pe, self.targets[idx]

    def __len__(self):
        return self.inputs.shape[0]
