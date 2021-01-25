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
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# RuntimeError: CUDA error: unspecified launch failure
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

n = 10  # 取前n天的資料作為特徵

#載入資料集
train_x = pd.read_csv(r'D:\dataset\lilium_price\train_x\100-108_2all.csv', encoding='utf-8')
train_y = pd.read_csv(r'D:\dataset\lilium_price\train_y\100-108_2all.csv', encoding='utf-8')
val_x = pd.read_csv(r'D:\dataset\lilium_price\val_x\100-108_2all.csv', encoding='utf-8')
val_y = pd.read_csv(r'D:\dataset\lilium_price\val_y\100-108_2all.csv', encoding='utf-8')

#正規化
x_scaler = StandardScaler().fit(train_x)
train_x = x_scaler.transform(train_x)
val_x = x_scaler.transform(val_x)

# to tensor
train_x = torch.Tensor(train_x)
train_y = torch.Tensor(np.array(train_y))
val_x = torch.Tensor(val_x)
val_y = torch.Tensor(np.array(val_y))
# Setloader
trainset = utils.Setloader(train_x, train_y)
valset = utils.Setloader(val_x, val_y)

# train
batch_size = 100
val_batch_size = 100
LR = 0.01
num_epochs = 100

model = model.RNN_modelv1(input_dim=train_x.shape[1], output_dim=train_y.shape[1]).to(device)
# 選擇優化器與損失函數
optimizer = torch.optim.AdamW(model.parameters(), lr=LR) 
criterion = nn.MSELoss().to(device)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
#                                                T_max=10,
#                                                eta_min=1e-6,
#                                                last_epoch=-1)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=True)
train_epoch_size = math.ceil(len(trainloader.dataset)/batch_size)
val_epoch_size = math.ceil(len(valloader.dataset)/val_batch_size)

loss_list = []
val_loss_list = []
mae_list = []
for epoch in range(num_epochs):
    epoch += 1
    print('running epoch: {} / {}'.format(epoch, num_epochs))
    #訓練模式
    model.train()
    total_loss = 0
    
    with tqdm(total=train_epoch_size) as pbar:
        for inputs, target in trainloader:
            inputs, target = inputs.to(device), target.to(device)
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
    total_mae = 0
    with tqdm(total=val_epoch_size) as pbar:
        with torch.no_grad():
            for inputs, target in valloader:
                inputs, target = inputs.to(device), target.to(device)
                output = model(torch.unsqueeze(inputs, dim=0))
                running_val_loss = criterion(torch.squeeze(output), target).item()
                running_mae = mean_absolute_error(target.cpu(), torch.squeeze(output).cpu())
                total_val_loss += running_val_loss*inputs.shape[0]
                total_mae += running_mae*inputs.shape[0]
                #更新進度條
                pbar.set_description('validation')
                pbar.set_postfix(
                        **{
                            'running_val_loss': running_val_loss,
                            'mae': running_mae
                        })
                pbar.update(1)
    scheduler.step()
    val_loss = total_val_loss/len(valloader.dataset)
    mae = total_mae/len(valloader.dataset)
    val_loss_list.append(val_loss)
    mae_list.append(mae)
    print('train_loss: {:.4f}, valid_loss: {:.4f}, MAE:{:.2f}, lr:{:.1e}'.format(loss, val_loss, mae, scheduler.get_last_lr()[0]) )
    #每10個epochs及最後一個epoch儲存模型
    if (not epoch % 10) or (epoch == num_epochs)  :
        torch.save(model.state_dict(), './logs/epoch%d-loss%.4f-val_loss%.4f-mae%.2f.pth' %(epoch, loss, val_loss, mae))


#繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss_list, label='Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend(loc='best')
plt.savefig('./images/loss.jpg')
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('mae')
plt.plot(mae_list)
plt.savefig('./images/mae.jpg')
plt.show()
