import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from net import model
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# #OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path_csv = r'D:\dataset\lilium_price\108\FS443.csv'
cloumn = ['最高價', '上價']
n = 5  # 取前n天的資料作為特徵
train_end = 200
train_df = pd.DataFrame()
val_df = pd.DataFrame()
for col in cloumn:
    train_col_df, val_col_df = utils.read_data(path_csv, col, n, train_end=train_end)    # shape = (train_end-n)*(n+1)
    train_df = pd.concat([train_df,train_col_df],axis=1)  
    val_df = pd.concat([val_df,val_col_df],axis=1)
# 正歸化
train = np.array(train_df)
val = np.array(val_df)
mean =  np.mean(train)
std = np.std(train)

train = (train -mean) / std
val = (val - mean) /std

# to tensor
train = torch.Tensor(train)
val = torch.Tensor(val)
trainset = utils.Setloader(train)
valnset = utils.Setloader(val)

# train
batch_size = 100
LR = 0.0001
num_epochs = 1000
model = model.RNN_model(n)
# 選擇優化器與損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
criterion = nn.MSELoss()

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valnset, batch_size=batch_size, shuffle=True)
train_epoch_size = math.ceil(len(trainloader.dataset)/batch_size)
val_epoch_size = math.ceil(len(valloader.dataset)/batch_size)

loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    epoch += 1
    print('running epoch: {} / {}'.format(epoch, num_epochs))
    #訓練模式
    model.train()
    total_loss = 0
    
    with tqdm(total=train_epoch_size) as pbar:
        for inputs, target in trainloader:
            output = model(torch.unsqueeze(inputs, dim=0))
            loss = criterion(torch.squeeze(output), target)
            running_loss = loss.item()
            total_loss += running_loss*inputs.shape[0]
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()            
            #更新進度條
            pbar.set_description('train')
            pbar.set_postfix(
                    **{
                        'running_loss': running_loss,
                    })
            pbar.update(1)
    loss = total_loss/len(trainloader.dataset)
    loss_list.append(loss)
    #評估模式
    model.eval()
    total_val_loss = 0
   
    with tqdm(total=val_epoch_size) as pbar:
        with torch.no_grad():
            for inputs, target in valloader:
                output = model(torch.unsqueeze(inputs, dim=0))
                loss = criterion(torch.squeeze(output), target)
                running_val_loss = loss.item()
                total_val_loss += running_val_loss*inputs.shape[0]
                #更新進度條
                pbar.set_description('validation')
                pbar.set_postfix(
                        **{
                            'running_val_loss': running_val_loss,
                        })
                pbar.update(1)
    val_loss = total_val_loss/len(valloader.dataset)
    val_loss_list.append(val_loss)
    print('train_loss: {}, valid_loss: {}'.format(loss, val_loss) )
    #每10個epochs及最後一個epoch儲存模型
    if (not epoch % 10) or (epoch == num_epochs)  :
        torch.save(model.state_dict(), './logs/epoch%d-loss%d-val_loss%.4f.pth' %
        (epoch, loss, val_loss))
print('mean: %.4f std: %.4f'%(mean,std))

#繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss_list, label='Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend(loc='best')
plt.savefig('./images/loss.jpg')
plt.show()