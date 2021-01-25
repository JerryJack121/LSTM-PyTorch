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
weight_list = ['FS443\epoch370-loss30295.4707-val_loss17190.6680-mae80.60.pth',
                'FS479\epoch962-loss5045.2241-val_loss12625.3564-mae61.74.pth',
                'FS592\epoch255-loss5028.7485-val_loss6465.3521-mae48.77.pth',
                'FS609\epoch499-loss61752.2070-val_loss11638.4756-mae57.40.pth',
                'FS639\epoch101-loss14279.4297-val_loss10592.9736-mae59.38.pth',
                'FS779\epoch100-loss36009.6602-val_loss8503.8037-mae52.10.pth',
                'FS859\epoch87-loss11530.5078-val_loss9995.5928-mae48.90.pth',
                'FS879\epoch27-loss20902.0957-val_loss18969.6211-mae73.45.pth',
                'FS899\epoch71-loss61343.3125-val_loss8101.2671-mae49.79.pth',
                'FS929\epoch18-loss24372.5078-val_loss36938.5195-mae92.60.pth']

n = 10  # 取前n天的資料作為特徵
f = 10  #花的種類數
p = 4   #預測的價格數量
model = model.RNN_modelv1(input_dim=n*p, output_dim=p)
for flower in flower_name:
    path_test = os.path.join(r'D:\dataset\lilium_price\test_x\for2020test', flower + '.csv')    #前一年的訓練資料
    path_weight = os.path.join(r'./weights', weight_list[flower_name.index(flower)]) #權重
    sub_path = './results/test_flower_price.csv'    #submit格式
    path_result_csv = os.path.join('./results', flower+'_result.csv')
    header = []
    for col in cloumn:
        header.append(col)
    test_df = pd.read_csv(path_test, index_col=None, header=None) 
    test_x = np.array(test_df)

    # # 正歸化
    train_x = pd.read_csv(os.path.join(r'D:\dataset\lilium_price\train_x', flower+'.csv'), encoding='utf-8')   #訓練時的資料用於正規化
    x_scaler = StandardScaler().fit(train_x)
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)  # to tensor

    model.load_state_dict(torch.load(path_weight))
    model.eval()
    date_list = []
    output_list = []
    sub = pd.read_csv(sub_path, header=1)
    for i in tqdm(range(len(sub)//f)):
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

