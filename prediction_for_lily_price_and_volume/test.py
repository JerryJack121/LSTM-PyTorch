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

test_csv = r'D:\dataset\lilium_price\109\FS443.csv'
path_lastyear_year = r'D:\dataset\lilium_price\108\FS443.csv'
path_weight = r'./weights/0122/epoch1000-loss0-val_loss0.4937.pth'
path_result_csv = './results/109result.csv'
cloumn = [ '上價', '中價', '平均價', '交易量']
n = 5  # 取前n天的資料作為特徵


test_date = pd.read_csv(test_csv, encoding='utf-8')['日　　期']
test_df = utils.read_col_data(test_csv, cloumn, n, path_lastyear_csv=path_lastyear_year)
test = np.array(test_df)

# 正歸化
train_df, val_df = utils.read_col_data(path_lastyear_year, cloumn, n , train_end=367)
train_x, train_y = utils.split_xy(train_df, len(cloumn), n)
x_scaler = StandardScaler().fit(train_x)
y_scaler = StandardScaler().fit(train_y)
test = x_scaler.transform(test)

test = torch.Tensor(test)  # to tensor
testloader = DataLoader(test, batch_size=1, shuffle=False)

model = model.RNN_model(input_dim=n*len(cloumn), output_dim=len(cloumn))
model.load_state_dict(torch.load(path_weight))

output_list = []
with tqdm(total=len(testloader.dataset)) as pbar:
    with torch.no_grad():
        for inputs in testloader:
            output = model(torch.unsqueeze(inputs, dim=0))
            output = np.array(output[0][0])
            output_list.append(output)
            # output_arr = np.vstack(output_arr, output)
            pbar.update(1)
output_arr = np.array(output_list)
output_arr = y_scaler.inverse_transform(output_arr)

#匯出csv
result_df = pd.DataFrame({
    '日期': test_date,
    '最高價':output_arr[:, 0],
    '上價':output_arr[:, 1],
    '平均價':output_arr[:, 2],
    '交易量':output_arr[:, 3]
}).to_csv(path_result_csv, index=None, encoding='utf_8_sig')
