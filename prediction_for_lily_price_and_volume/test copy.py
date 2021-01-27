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
weight_list = [r'FS443.pth',
                r'FS479.pth',
                r'FS592.pth',
                r'FS609.pth',
                r'FS639.pth',
                r'FS779.pth',
                r'FS859.pth',
                r'FS879.pth',
                r'FS899.pth',
                r'FS929.pth']

n = 10  # 取前n天的資料作為特徵
f = 10  #花的種類數
p = 4   #預測的價格數量
model = model.RNN(input_size=n*p, output_size=p)
for flower in flower_name:
    path_test = os.path.join(r'D:\dataset\lilium_price\test_x\for2020test', flower + '.csv')    #載入測試初始值
    path_weight = os.path.join(r'E:\Blacky_Lily', weight_list[flower_name.index(flower)]) #權重
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
    sub = pd.read_csv(sub_path, header=1)
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
    for i in tqdm(range(len(sub[130:])//f+1)):
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

