#載入市價資料
import pandas as pd
import os
from tqdm import tqdm

io = r'D:\dataset\lilium_price\108\FS443.xls'
df = pd.read_excel(io, header=4)


with tqdm(total=len(df)) as pbar:
    for index, row in df.iterrows():
        data_row = df.loc[index].values[:-1]
        # print(data_row)
        date, market, product, highest_price, hp, mp, lp, ap, _, sales_volume,_,_ = data_row
        print('\n')
        # 更新進度條
        pbar.update(1)
        pbar.set_description('load')
        pbar.set_postfix(**{
            'date': date,
        })