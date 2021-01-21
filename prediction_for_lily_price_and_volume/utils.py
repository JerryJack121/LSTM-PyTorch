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
def read_data(path_csv, cloumn, n , path_lastyear_csv=None, train_end=None):
    df = pd.read_csv(path_csv, encoding='utf-8')    
    df_col = df[cloumn].astype(float)
    if path_lastyear_csv:   #用於產生測試資料集
        lastyear_df = pd.read_csv(path_lastyear_csv, encoding='utf-8')[-n:]
        last_df_col = lastyear_df[cloumn].astype(float)
        df_col = pd.concat([last_df_col,df_col],axis=0, ignore_index=True)
        print(df_col)
        test_df = generate_df_affect_by_n_days(cloumn, df_col, n, mode ='test')
        return test_df
    train_series, test_series = df_col[:train_end], df_col[train_end - n:]
    train_df = generate_df_affect_by_n_days(cloumn, train_series, n, mode='train')
    test_df = generate_df_affect_by_n_days(cloumn, test_series, n, mode='valid')

    return train_df, test_df

class Setloader(Dataset):
    def __init__(self, data, n , num_col):
        # self.data, self.label = data[:, :-1].float(), data[:, -1].float()
        for i in range(num_col):
            if i == 0:
                all_data =  data[:, 0:n].float()
                all_label = data[:, n].float()

            else:
                # print(data[:, ((n+1)*i):(n*i+1)].float())
                all_data = np.concatenate((all_data, data[:, (n+1)*i:n*(i+1)+1].float()), axis=1)
                all_label =np.vstack((all_label,all_label, data[:, (n+1)*(i+1)-1].float()))
        print(all_label)
        self.data = all_data
        self.label = all_label
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    



