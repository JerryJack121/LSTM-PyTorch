import pandas as pd
import numpy as np

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
    df_col = df[cloumn][:50].astype(float)
    train_series, test_series = df_col[:train_end], df_col[train_end - n:]
    train_df = generate_df_affect_by_n_days(train_series, n)

    return train_df

if __name__ == "__main__":
    path_csv = r'D:\dataset\lilium_price\108\FS443.csv'
    cloumn = '最高價'
    n = 5  #取前n天的資料作為特徵
    train_end = 20
    train_df = read_data(path_csv, cloumn, n, train_end)    #train_df.shape = (train_end-n)*(n+1)


#正歸化
# train = (train - np.mean(train)) / np.std(train)

