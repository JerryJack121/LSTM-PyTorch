import pandas as pd
import numpy as np
from torch.utils.data import Dataset

#將前n天的資料作為訓練特徵，當天的資料作為Label。
def generate_df_affect_by_n_days(series, n):
    df = pd.DataFrame()
    for i in range(n):
        df['d%d'%i] = series.tolist()[i:-(n - i)]    #tolist解決index的問題
    df['label'] = series.tolist()[n:]

    return df

#載入資料集
def read_data(path_csv, cloumn,n ,train_end):
    df = pd.read_csv(path_csv, encoding='utf-8')
    df_col = df[cloumn].astype(float)
    train_series, test_series = df_col[:train_end], df_col[train_end - n:]
    train_df = generate_df_affect_by_n_days(train_series, n)

    return train_df

class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    



