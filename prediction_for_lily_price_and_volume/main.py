import pandas as pd
import numpy as np
import torch
import utils
from torch.utils.data import Dataset, DataLoader

path_csv = r'D:\dataset\lilium_price\108\FS443.csv'
cloumn = '最高價'
n = 5  # 取前n天的資料作為特徵
train_end = 20

train_df = utils.read_data(path_csv, cloumn, n, train_end)    # shape = (train_end-n)*(n+1)

# 正歸化
train = np.array(train_df)
train = (train - np.mean(train)) / np.std(train)
# to tensor
train = torch.Tensor(train)

trainset = utils.TrainSet(train)
trainloader = DataLoader(trainset, batch_size=10, shuffle=False)
