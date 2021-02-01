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

def new_x(predict, test_x, n, x_scaler):
    test_x =  x_scaler.inverse_transform(test_x)

    for i in range(test_x.shape[1]//n):
        test_x = np.insert(test_x, int(i*n+n), values=float(predict[i]), axis=1)
        test_x = np.delete(test_x, i*n, axis=1)
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)
    return test_x


n = 10  # 取前n天的資料作為特徵
f = 10  #花的種類數
p = 6   #預測的價格數量


path_weight = r'E:\Blacky_Lily\epoch2574_train_loss_35653.3867_val_loss_15754.0000_mae_69.4205.pth' #權重
sub_path = './results/test_flower_price.csv'    #submit格式
path_result_csv = './results/0127result.csv'
flower_name = ['FS443', 'FS479', 'FS592', 'FS609', 'FS639', 'FS779', 'FS859', 'FS879', 'FS899', 'FS929']    # 需與訓練時的處理順序相同
cloumn = ['mp', 'up', 'mp', 'lp', 'ap', 's']    # 需與訓練時的處理順序相同
model = model.RNN(input_size=n*p*f, output_size=p*f)
model.load_state_dict(torch.load(path_weight))

header = ['date']
for flower in flower_name:
    for col in cloumn:
        header.append('%s_%s'%(flower, col))
sub = pd.read_csv(sub_path)
# # 正歸化
train_x = pd.read_csv(r'D:\dataset\lilium_price\train_data.csv', index_col=0, encoding='utf-8')
x_scaler = StandardScaler().fit(train_x)


test_df = pd.read_csv(r'D:\dataset\lilium_price\test_x\for2020test\for2020test.csv', index_col=None, header=None)  # 取去年的最後n天作為今天的初始特徵
test_x = np.array(test_df)
test_x = x_scaler.transform(test_x)
test_x = torch.Tensor(test_x)  # to tensor


model.eval()
date_list = []
output_list = []
start = datetime.datetime(2020, 1,9)
end = datetime.datetime(2020, 1,24)
interval = (end - start).days
for i in tqdm(range(interval)):
    with torch.no_grad():
        date = start + datetime.timedelta(days = (i + 1))
        date = date.strftime('%Y-%m-%d')
        date_list.append(date)
        predict = model(torch.unsqueeze(test_x, dim=0))
        predict = predict[0][0]    
        test_x = new_x(predict, test_x, n, x_scaler)
        predict = np.array(predict)
        output_list.append(predict)

test_df = pd.read_csv(r'D:\dataset\lilium_price\test_x\for2021test\for2021test.csv', index_col=None, header=None)  # 取去年的最後n天作為今天的初始特徵
test_x = np.array(test_df)
test_x = x_scaler.transform(test_x)
test_x = torch.Tensor(test_x)  # to tensor
start = datetime.datetime(2021, 1,27)
end = datetime.datetime(2021, 2,12)
interval = (end - start).days
for i in tqdm(range(interval)):
    with torch.no_grad():
        date = start + datetime.timedelta(days = (i + 1))
        date = date.strftime('%Y-%m-%d')
        date_list.append(date)
        predict = model(torch.unsqueeze(test_x, dim=0))
        predict = predict[0][0]    
        test_x = new_x(predict, test_x, n, x_scaler)
        predict = np.array(predict)
        output_list.append(predict)




output_arr = np.array(output_list)
result_df = pd.DataFrame(output_arr)
result_df.insert(0,'date',date_list)

        
result_df.to_csv(path_result_csv, float_format='%.2f', header=header, encoding='utf_8')
result_df = pd.read_csv(path_result_csv, encoding='utf_8')

for flower in flower_name:
    for col in ['mp', 'lp']:
        result_df.drop(flower+'_'+col, axis=1)
result_df.to_csv(path_result_csv, float_format='%.2f',index=None, encoding='utf_8')


# #輸出預測結果
result_df = pd.read_csv(path_result_csv, index_col=0, encoding='utf_8')
# print(result_df)
for index, row in sub.iterrows(): 
    date = row['date']
    flower = row['flower_no']
    try:
        x = result_df[result_df.date == date].index
        y = 4*(flower_name.index(flower))
        sub.iloc[index, 3:7] = result_df.iloc[x, y+1:y+4+1].values[0]
    # print(sub.iloc[index])
    except:
        print(date)
        continue
    
sub.to_csv('./results/0127-7_submit.csv', float_format='%.1f', index=None)