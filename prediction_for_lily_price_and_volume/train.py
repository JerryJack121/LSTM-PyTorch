import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from net import model
import math
from tqdm import tqdm

path_csv = r'D:\dataset\lilium_price\108\FS443.csv'
cloumn = '最高價'
n = 5  # 取前n天的資料作為特徵
train_end = 200

train_df = utils.read_data(path_csv, cloumn, n, train_end)    # shape = (train_end-n)*(n+1)

# 正歸化
train = np.array(train_df)
# train = (train - np.mean(train)) / np.std(train)
# to tensor
train = torch.Tensor(train)
trainset = utils.TrainSet(train)


#train
batch_size = 10
LR = 0.0001
num_epochs = 100
rnn = model.RNN(n)
# 選擇優化器與損失函數
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) 
criterion = nn.MSELoss()

trainloader = DataLoader(trainset, batch_size=10, shuffle=False)
train_epoch_size = math.ceil(train.shape[0]/batch_size)

for epoch in range(num_epochs):
    print('running epoch: {} / {}'.format(epoch, num_epochs))
    with tqdm(total=train_epoch_size) as pbar:
        for inputs, target in trainloader:
            output = rnn(torch.unsqueeze(inputs, dim=0))
            loss = criterion(torch.squeeze(output), target)
            running_loss = loss.item()
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
            
            #更新進度條
            pbar.set_postfix(
                    **{
                        'running_loss': running_loss,
                    })
            pbar.update(1)

    # if step % 10:
    #     torch.save(rnn, 'rnn.pkl')