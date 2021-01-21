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

train_df, val_df = utils.read_data(path_csv, cloumn, n, train_end)    # shape = (train_end-n)*(n+1)

# 正歸化
train = np.array(train_df)
val = np.array(val_df)
# train = (train - np.mean(train)) / np.std(train)
# val = (val - np.mean(train)) / np.std(train)

# to tensor
train = torch.Tensor(train)
val = torch.Tensor(val)
trainset = utils.Setloader(train)
valnset = utils.Setloader(val)


# train
batch_size = 10
LR = 0.0001
num_epochs = 11
rnn = model.RNN(n)
# 選擇優化器與損失函數
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) 
criterion = nn.MSELoss()

trainloader = DataLoader(trainset, batch_size=10, shuffle=False)
valloader = DataLoader(valnset, batch_size=10, shuffle=False)
train_epoch_size = math.ceil(train.shape[0]/batch_size)
val_epoch_size = math.ceil(val.shape[0]/batch_size)


for epoch in range(num_epochs):
    epoch += 1
    print('running epoch: {} / {}'.format(epoch, num_epochs))
    #訓練模式
    rnn.train()
    total_loss = 0
    with tqdm(total=train_epoch_size) as pbar:
        for inputs, target in trainloader:
            output = rnn(torch.unsqueeze(inputs, dim=0))
            loss = criterion(torch.squeeze(output), target)
            running_loss = loss.item()
            total_loss += running_loss/inputs.shape[0]
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()            
            #更新進度條
            pbar.set_postfix(
                    **{
                        'running_loss': running_loss,
                    })
            pbar.update(1)
    #評估模式
    rnn.eval()
    total_val_loss = 0
    with tqdm(total=val_epoch_size) as pbar:
        with torch.no_grad():
            for inputs, target in valloader:
                output = rnn(torch.unsqueeze(inputs, dim=0))
                loss = criterion(torch.squeeze(output), target)
                running_val_loss = loss.item()
                total_val_loss += running_val_loss/inputs.shape[0]
                #更新進度條
                pbar.set_postfix(
                        **{
                            'running_val_loss': running_val_loss,
                        })
                pbar.update(1)
    print('train_loss: {}, valid_loss: {}'.format(total_loss, total_val_loss) )
    #每10個epochs及最後一個epoch儲存模型
    if (not epoch % 10) or (epoch == num_epochs)  :
        torch.save(rnn.state_dict(), './logs/epoch%d-loss%d-val_loss%.4f.pth' %
        (epoch, total_loss, total_val_loss))
