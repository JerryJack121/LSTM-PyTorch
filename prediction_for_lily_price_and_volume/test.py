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


test_csv = r'D:\dataset\lilium_price\109\FS443.csv'
path_lastyear_year = r'D:\dataset\lilium_price\val_x\108all.csv'
path_weight = r'./weights/epoch50-loss0.6024-val_loss0.6899.pth'
path_result_csv = './results/109result.csv'
cloumn = [ '上價', '中價', '平均價', '交易量']
n = 5  # 取前n天的資料作為特徵


test_date = pd.read_csv(test_csv, encoding='utf-8')['日　　期']
test_df = pd.read_csv(path_lastyear_year)[-1:]  # 取去年的最後n天作為今天的初始特徵
test_x = np.array(test_df)

# # 正歸化
train_x = pd.read_csv(r'D:\dataset\lilium_price\train_x\108all.csv', encoding='utf-8')
x_scaler = StandardScaler().fit(train_x)
test_x = x_scaler.transform(test_x)
test_x = torch.Tensor(test_x)  # to tensor

model = model.RNN_model(input_dim=200, output_dim=40)
model.load_state_dict(torch.load(path_weight))

output_list = []
with tqdm(total=372) as pbar:
    for i in range(372):
        with torch.no_grad():
            predict = model(torch.unsqueeze(test_x, dim=0))
            predict = predict[0][0]     
            test_x = new_x(predict, test_x, n, x_scaler)
            output_list.append(np.array(predict))
            pbar.update(1)
output_arr = np.array(output_list)

#匯出csv
result_df = pd.DataFrame({
    '日期': test_date,
    '最高價':output_arr[:, 0],
    '上價':output_arr[:, 1],
    '平均價':output_arr[:, 2],
    '交易量':output_arr[:, 3]
}).to_csv(path_result_csv, index=None, encoding='utf_8_sig')
