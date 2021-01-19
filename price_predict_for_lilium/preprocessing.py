#載入市價資料
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

org_path = r'D:\dataset\lilium_price\org\108\FS443.xls' #原始資料
path = r'D:\dataset\lilium_price\108\FS443.csv' #生成資料
df = pd.read_excel(org_path, header=4)

# x = df.loc[df['日　　期'] == '108/01/10'] #查詢特定日期

i = 0

with tqdm(total=len(df)) as pbar:
    df = df.drop(['增減%', '交易量', '增減%.1', '殘貨量', 'Unnamed: 12'], axis=1)
    for index, row in df.iterrows():
        data_row = df.loc[index].values
        # print(data_row)
        date, market, product, highest_price, hp, mp, lp, ap= data_row    #日期、市場、產品、最高價、上價、中價、下價、平均價、增減%、交易量、增減%、殘貨量
        # print(date)
        if date == '小　　計':
            break
        month = date.split('/')[1]
        date = date.split('/')[2]
        # print(month)
        # print(date)
        # print((int(month)-1)*31 + int(date))  
        date = (int(month)-1)*31 + int(date)    
        df.loc[index, '日　　期'] = date
        # 更新進度條
        pbar.update(1)
        pbar.set_description('preprocessing')
        pbar.set_postfix(**{
            # 'date': date,
        })
df.to_csv(path, encoding='utf_8_sig', index=False)