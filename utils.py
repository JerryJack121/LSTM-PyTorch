import pandas as pd
import numpy as np
from torch.utils.data import Dataset

#將前n天的資料作為訓練特徵，當天的資料作為Label。
def generate_df_affect_by_n_days(col_name, series, n, mode):
    df = pd.DataFrame()
    for i in range(n):
        df['%s_d%d'%(col_name, i)] = series.tolist()[i:-(n - i)]    #tolist解決index的問題
    if not mode == 'test':
        df['y_%s'%col_name] = series.tolist()[n:]

    return df

#載入資料集
def read_data(path_csv, cloumn, n , path_lastyear_csv, train_end):
    df = pd.read_csv(path_csv, encoding='utf-8')    
    df_col = df[cloumn].astype(float)
    if path_lastyear_csv:   #用於產生測試資料集
        lastyear_df = pd.read_csv(path_lastyear_csv, encoding='utf-8')[-n:]
        last_df_col = lastyear_df[cloumn].astype(float)
        df_col = pd.concat([last_df_col,df_col],axis=0, ignore_index=True)
        test_df = generate_df_affect_by_n_days(cloumn, df_col, n, mode ='test')
        return test_df
    train_series, test_series = df_col[:train_end], df_col[train_end - n:]
    train_df = generate_df_affect_by_n_days(cloumn, train_series, n, mode='train')
    test_df = generate_df_affect_by_n_days(cloumn, test_series, n, mode='valid')

    return train_df, test_df

def read_col_data(path_csv, cloumn, n , path_lastyear_csv=None, train_end=None):
    if train_end:   #訓練、驗證集
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        for col in cloumn:           
                train_col_df, val_col_df = read_data(path_csv, col, n, path_lastyear_csv, train_end)    # shape = (train_end-n)*(n+1)
                train_df = pd.concat([train_df,train_col_df],axis=1)  
                val_df = pd.concat([val_df,val_col_df],axis=1)
        return train_df, val_df
    else:   #測試集
        test_df = pd.DataFrame()
        for col in cloumn:           
                test_col_df = read_data(path_csv, col, n, path_lastyear_csv, train_end)    # shape = (train_end-n)*(n+1)
                test_df = pd.concat([test_df,test_col_df],axis=1)  
        return test_df

# 切分特徵、label
def split_xy(df, num_col, n):
    arr = np.array(df)
    for i in range(num_col):
        if i == 0:
            x =  arr[:, 0:n]
            y = arr[:, n].reshape(-1,1)
        else:
            x = np.concatenate((x, arr[:, (n+1)*i:(n+1)*(i+1)-1]), axis=1)
            y = np.concatenate((y, arr[:, (n+1)*(i+1)-1].reshape(-1,1)), axis=1)

    return x, y

class Setloader(Dataset):
    def __init__(self, data, label):
        # self.data, self.label = data[:, :-1].float(), data[:, -1].float()
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    



