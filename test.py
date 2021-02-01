import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from net import model
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import datetime
import os

def new_x(predict, test_x, n, x_scaler):
    test_x =  x_scaler.inverse_transform(test_x)
    for i in range(test_x.shape[1]//n):
        test_x = np.insert(test_x, int(i*n+n), values=float(predict[i]), axis=1)
        test_x = np.delete(test_x, i*n, axis=1)
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)
    return test_x

# start = datetime.datetime(2019, 12,31)
# end = datetime.datetime(2020, 12,31)
# interval = (end - start).days


flower_name = ['FS443', 'FS479', 'FS592', 'FS609', 'FS639', 'FS779', 'FS859', 'FS879', 'FS899', 'FS929']
cloumn = [ 'price_high', 'price_mid', 'price_avg', 'volume']    # 需與訓練時的處理順序相同
weight_list = [r'FS443\epoch3009-loss5322.0649-val_loss11729.3789-mae61.65.pth',
                r'FS479\epoch3134-loss10813.8223-val_loss21071.4609-mae68.87.pth',
                r'FS592\epoch181-loss9081.8652-val_loss5275.4360-mae47.46.pth',
                r'FS609\epoch599-loss67233.6875-val_loss12579.5771-mae58.82.pth',
                r'FS639\epoch382-loss10563.7393-val_loss55081.4414-mae72.72.pth',
                r'FS779\epoch354-loss35233.5977-val_loss18121.4668-mae56.62.pth',
                r'FS859\epoch13-loss72137.3984-val_loss10614.0059-mae56.64.pth',
                r'FS879\epoch960-loss6619.4561-val_loss86955.9375-mae96.01.pth',
                r'FS899\epoch898-loss20553.2910-val_loss34573.2227-mae72.24.pth',
                r'FS929\epoch898-loss20553.2910-val_loss34573.2227-mae72.24.pth']

n = 10  # 取前n天的資料作為特徵
f = 10  #花的種類數
p = 4   #預測的價格數量
model = model.RNN_modelv1(input_dim=n*p, output_dim=p)
for flower in flower_name:
    path_test = os.path.join(r'D:\dataset\lilium_price\test_x\for2020test', flower + '.csv')    #載入測試初始值
    path_weight = os.path.join(r'.\weights', weight_list[flower_name.index(flower)]) #權重
    sub_path = './results/test_flower_price.csv'    #submit格式
    path_result_csv = os.path.join('./results', flower+'_result.csv')
    header = []
    for col in cloumn:
        header.append(col)
    test_df = pd.read_csv(path_test, index_col=None, header=None) 
    test_x = np.array(test_df)

    # 正歸化
    train_x = pd.read_csv(os.path.join(r'D:\dataset\lilium_price\train_x', flower+'.csv'), encoding='utf-8')   #訓練時的資料用於正規化
    x_scaler = StandardScaler().fit(train_x)
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)  # to tensor

    model.load_state_dict(torch.load(path_weight))
    model.eval()
    date_list = []
    output_list = []
    sub = pd.read_csv(sub_path, header=0)
    for i in tqdm(range(len(sub[:130])//f)):
        with torch.no_grad():
            predict = model(torch.unsqueeze(test_x, dim=0))
            predict = predict[0][0]    
            test_x = new_x(predict, test_x, n, x_scaler)
            predict = np.array(predict)
            output_list.append(predict)

    path_test = os.path.join(r'D:\dataset\lilium_price\test_x\for2021test', flower + '.csv')    #載入測試初始值
    test_df = pd.read_csv(path_test, index_col=None, header=None) 
    test_x = np.array(test_df)
    # 正歸化
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)  # to tensor
    for i in tqdm(range(len(sub[130:])//f)):
        with torch.no_grad():
            predict = model(torch.unsqueeze(test_x, dim=0))
            predict = predict[0][0]    
            test_x = new_x(predict, test_x, n, x_scaler)
            predict = np.array(predict)
            output_list.append(predict)


    output_arr = np.array(output_list)
    result_df = pd.DataFrame(output_arr)
    # result_df.insert(0,'date',sub['date'])
    result_df.to_csv(path_result_csv, float_format='%.2f', header=header, index=None)
    print(flower,' Done')

