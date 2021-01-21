import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from net import model
import math
from tqdm import tqdm

path_csv = r'D:\dataset\lilium_price\109\FS443.csv'
path_lastyear_year = r'D:\dataset\lilium_price\108\FS443.csv'
path_result_csv = './results/result.csv'
cloumn = '最高價'
n = 5  # 取前n天的資料作為特徵
path_weight = r'./weights/epoch1000-loss0-val_loss0.4433.pth'

test_date = pd.read_csv(path_csv, encoding='utf-8')['日　　期']
test_df = utils.read_data(path_csv, cloumn, n, path_lastyear_year)

# 正歸化
test = np.array(test_df)
mean = 208.5607
std = 53.5782
test = (test - mean) / std

test = torch.Tensor(test)  # to tensor
testloader = DataLoader(test, batch_size=1, shuffle=False)

model = model.RNN_model(n)
model.load_state_dict(torch.load(path_weight))

output_list = []
with tqdm(total=len(testloader.dataset)) as pbar:
    with torch.no_grad():
        for inputs in testloader:
            output = model(torch.unsqueeze(inputs, dim=0))
            output = int(output * std + mean)
            output_list.append(output)
            pbar.update(1)

result = pd.DataFrame({
    '日期': test_date,
    cloumn: output_list
}).to_csv(path_result_csv, index=None, encoding='utf_8_sig')
