import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from net import model

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

#train
LR = 0.0001
EPOCH = 100

rnn = model.RNN(n)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        print(output)
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
        running_loss = loss.item()
    print(step, running_loss)
    # if step % 10:
    #     torch.save(rnn, 'rnn.pkl')