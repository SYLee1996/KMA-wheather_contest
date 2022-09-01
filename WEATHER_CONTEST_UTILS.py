import os 
import math
import random
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm 

import torch

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Custom_Dataset(Dataset):
    def __init__(self, X, Y, config):
        self.X = X
        self.Y = Y
        self.mode = config['mode']
        self.augmentation = config['augmentation']
        
        x_columns = list(X.columns)
        sub_list = ['band1', 'band2', 'band3', 'band4', 'band5',
                    'band6', 'band7', 'band8', 'band9', 'band10', 'band11', 'band12',
                    'band13', 'band14', 'band15', 'band16', 'solarza', 'esr']
        self.columns = list(set(x_columns) & set(sub_list))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        data = self.X.iloc[i]
        
        if (self.mode == 'TRAIN') | (self.mode == 'VALID'):
            if self.augmentation==True:
                augmentation = random.random()
                
                if 0.5 <= augmentation:
                    data[self.columns] = data[self.columns] * random.uniform(1, 0.95)
                elif  0.5 >= augmentation:
                    data = data
            
            return {
                'x' : torch.tensor(data, dtype=torch.float32),
                'y' : torch.tensor(self.Y.iloc[i], dtype=torch.float32),
            }
        else:
            return {
                'x' : torch.tensor(data, dtype=torch.float32),
            }
        

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                

# Data load & return concated dataframe
def load_data(folder_path, mode):
    files = os.listdir(folder_path)
    path_list = []
    
    # 파일 불러와서, 하나의 데이터 프레임으로 합치기
    bin_df = pd.DataFrame()

    for file in tqdm(files):
        if mode == 'TRAIN':
            if ('uv' in file)&('2022' not in file): 
                file_path = os.path.join(folder_path, file)
                path_list.append(file_path)
        else:
            if ('uv' in file)&('202206_uv' in file): 
                file_path = os.path.join(folder_path, file)
                path_list.append(file_path)
                
    if mode == 'TRAIN':
        for df_path in path_list:
            file_df = pd.read_csv(df_path, index_col=0)
            file_df.columns=["yyyymmdd", "hhnn", "stn", 
                            "lon", "lat", "uv", 
                            "band1", "band2", "band3", "band4", "band5", "band6",
                            "band7", "band8", "band9", "band10", "band11", "band12",
                            "band13", "band14", "band15", "band16", 
                            "solarza", "sateza", "esr", "height", "landtype"]
            bin_df = pd.concat([bin_df, file_df], axis=0)
    else:
        for df_path in path_list:
            bin_df = pd.read_csv(df_path, index_col=0)
            bin_df.columns=[ "stn", 
                            "lon", "lat", "uv", 
                            "band1", "band2", "band3", "band4", "band5", "band6",
                            "band7", "band8", "band9", "band10", "band11", "band12",
                            "band13", "band14", "band15", "band16", 
                            "solarza", "sateza", "esr", "height", "landtype"]
    return bin_df


# hhnn 변수의 형태 맞추는 함수
def trans_func(x):
    if len(str(x)) == 1:
        return '000'+str(x)
    elif len(str(x)) == 2:
        return '00'+str(x)
    elif len(str(x)) == 3:
        return '0'+str(x)
    else:
        return str(x)


# 특정 영역 구하는 함수
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


# cyclical embedding
def dummy_and_add_feature(x):
    minute = int(x.split(" ")[-1].split(":")[1])
    hour = int(x.split(" ")[-1].split(":")[0])
    day = int(x.split("-")[-1].split(" ")[0])
    month = int(x.split("-")[1])
    
    sin_minute = np.sin((2*np.pi*minute*60)/(24*60))
    cos_minute = np.cos((2*np.pi*minute*60)/(24*60))
    sin_hour = np.sin((2*np.pi*hour*60*60)/(24*60*60))
    cos_hour = np.cos((2*np.pi*hour*60*60)/(24*60*60))
    sin_day = np.sin((2*np.pi*day*24*60*60)/(31*24*60*60))
    cos_day = np.cos((2*np.pi*day*24*60*60)/(31*24*60*60))
    sin_month = np.sin((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    cos_month = np.cos((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    
    return sin_minute, cos_minute, sin_hour, cos_hour, sin_day, cos_day, sin_month, cos_month