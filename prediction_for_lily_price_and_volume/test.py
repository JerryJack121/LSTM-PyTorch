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
id = np.arange(1, 373)
def new_x(predict, test_x, n, x_scaler):
    test_x =  x_scaler.inverse_transform(test_x)

    for i in range(test_x.shape[1]//n):
        test_x = np.insert(test_x, int(i*n+n), values=float(predict[i]), axis=1)
        test_x = np.delete(test_x, i*n, axis=1)
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)
    return test_x


path_lastyear_year = r'D:\dataset\lilium_price\val_x\108all.csv'    #前一年的訓練資料
path_weight = r'./weights/epoch5000-loss121831.8831-val_loss26875.4116.pth' #權重
sub_path = './results/test_flower_price.csv'    #submit格式
path_result_csv = './results/109submit.csv'
flower_name = ['FS443', 'FS479', 'FS592', 'FS609', 'FS639', 'FS779', 'FS859', 'FS879', 'FS899', 'FS929']    # 需與訓練時的處理順序相同
cloumn = [ 'price_high', 'price_mid', 'price_avg', 'volume']    # 需與訓練時的處理順序相同

n = 10  # 取前n天的資料作為特徵
f = 10  #花的種類數
p = 4   #預測的價格數量

test_df = pd.read_csv(path_lastyear_year)[-1:]  # 取去年的最後n天作為今天的初始特徵
test_x = np.array(test_df)

# # 正歸化
train_x = pd.read_csv(r'D:\dataset\lilium_price\train_x\108all.csv', encoding='utf-8')
x_scaler = StandardScaler().fit(train_x)
test_x = x_scaler.transform(test_x)
test_x = torch.Tensor(test_x)  # to tensor

model = model.RNN_model(input_dim=n*p*f, output_dim=p*f)
model.load_state_dict(torch.load(path_weight))
model.eval()

output_list = []
for i in tqdm(range(372)):
    with torch.no_grad():
        predict = model(torch.unsqueeze(test_x, dim=0))
        predict = predict[0][0]     
        test_x = new_x(predict, test_x, n, x_scaler)
        output_list.append(np.array(predict))
output_arr = np.array(output_list)

#輸出預測結果
sub = pd.read_csv(sub_path)
for index, row in sub.iterrows():
    date = row['date']
    flower = row['flower_no']
    month = int(date.split('-')[1])
    day = int(date.split('-')[2])
    date_id = month*31 + day - 1
    sub.loc[index, 'price_high'] = int(output_arr[date_id, p*(flower_name.index(flower))+0])
    sub.loc[index, 'price_mid'] = int(output_arr[date_id, p*(flower_name.index(flower))+1])
    sub.loc[index, 'price_avg'] = round((output_arr[date_id, p*(flower_name.index(flower))+2]), 1)
    sub.loc[index, 'volume'] = int(output_arr[date_id, p*(flower_name.index(flower))+3])
sub.to_csv(path_result_csv, index=None)