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

def new_x(predict, test_x, n, x_scaler):
    test_x =  x_scaler.inverse_transform(test_x)

    for i in range(test_x.shape[1]//n):
        test_x = np.insert(test_x, int(i*n+n), values=float(predict[i]), axis=1)
        test_x = np.delete(test_x, i*n, axis=1)
    test_x = x_scaler.transform(test_x)
    test_x = torch.Tensor(test_x)
    return test_x


path_lastyear_year = r'D:\dataset\lilium_price\val_x\108all.csv'
path_weight = r'./weights/epoch10000-loss33934.8047-val_loss33840.2803.pth'
path_result_csv = './results/predict.csv'
# cloumn = [ '上價', '中價', '平均價', '交易量']
n = 5  # 取前n天的資料作為特徵


test_df = pd.read_csv(path_lastyear_year)[-1:]  # 取去年的最後n天作為今天的初始特徵
test_x = np.array(test_df)

# # 正歸化
train_x = pd.read_csv(r'D:\dataset\lilium_price\train_x\108all.csv', encoding='utf-8')
x_scaler = StandardScaler().fit(train_x)
test_x = x_scaler.transform(test_x)
test_x = torch.Tensor(test_x)  # to tensor

model = model.RNN_model(input_dim=200, output_dim=40)
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

#匯出csv
pd.DataFrame(output_arr).to_csv(path_result_csv, encoding='utf_8_sig')
