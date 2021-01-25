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

start = datetime.datetime(2019, 12,31)
end = datetime.datetime(2020, 12,31)
interval = (end - start).days

path_test = r'D:\dataset\lilium_price\test_x\for2020test\FS479.csv'    #前一年的訓練資料
path_weight = r'./weights/FS479/epoch962-loss5045.2241-val_loss12625.3564-mae61.74.pth' #權重
sub_path = './results/test_flower_price.csv'    #submit格式
path_result_csv = './results/FS479_result.csv'
# flower_name = ['FS443', 'FS479', 'FS592', 'FS609', 'FS639', 'FS779', 'FS859', 'FS879', 'FS899', 'FS929']    # 需與訓練時的處理順序相同
cloumn = [ 'price_high', 'price_mid', 'price_avg', 'volume']    # 需與訓練時的處理順序相同

n = 10  # 取前n天的資料作為特徵
f = 10  #花的種類數
p = 4   #預測的價格數量

header = []
for col in cloumn:
    header.append(col)

test_df = pd.read_csv(path_test, index_col=None, header=None)  # 取去年的最後n天作為今天的初始特徵
test_x = np.array(test_df)

# # 正歸化
train_x = pd.read_csv(r'D:\dataset\lilium_price\train_x\FS479.csv', encoding='utf-8')   #訓練時的資料用於正規化
x_scaler = StandardScaler().fit(train_x)
test_x = x_scaler.transform(test_x)
test_x = torch.Tensor(test_x)  # to tensor

model = model.RNN_modelv1(input_dim=n*p, output_dim=p)
model.load_state_dict(torch.load(path_weight))
model.eval()
date_list = []
output_list = []
sub = pd.read_csv(sub_path)
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


# # #輸出預測結果
# sub = pd.read_csv(sub_path)
# for index, row in sub.iterrows():
#     date = row['date']
#     flower = row['flower_no']
#     try:
#         x = result_df[result_df.date == date].index
#         y = p*(flower_name.index(flower))
#         sub.iloc[index, 3:7] = result_df.iloc[x, y+1:y+p+1].values[0]
#         # print(sub.iloc[index])
#     except:
#         print(date)
#         break
    
# sub.to_csv('./results/109_submit.csv', float_format='%.1f', index=None)