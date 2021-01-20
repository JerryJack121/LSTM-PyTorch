#載入市價資料
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

io = r'D:\dataset\lilium_price\108\FS443.xls'
df = pd.read_excel(io, header=4)

# x = df.loc[df['日　　期'] == '108/01/10'] #查詢特定日期

i = 0

with tqdm(total=len(df)) as pbar:
    for index, row in df[i:i+5].iterrows():
        price_list = []
        data_row = df.loc[index].values[:-1]
        # print(data_row)
        date, market, product, highest_price, hp, mp, lp, ap, _, _,_,_ = data_row    #日期、市場、產品、最高價、上價、中價、下價、平均價、增減%、交易量、增減%、殘貨量
        print('\n')
        # 更新進度條
        pbar.update(1)
        pbar.set_description('load')
        pbar.set_postfix(**{
            'date': date,
        })