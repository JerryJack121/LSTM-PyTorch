#載入市價資料
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

#填補空缺日期
def fill_df(df, add_date_list, sum_highest_price, sum_price_high, sum_price_mid, sum_price_low, sum_price_avg, sum_volume):
    id = len(df) - 1    #原始資料最後一列為'小計'，實際長度為len(df) - 1
    #取個欄平均
    avg_highest_price = round((sum_highest_price / id), 2)  #四捨五入制小數點後2位
    avg_price_high = round((sum_price_high / id), 2)
    avg_price_mid = round((sum_price_mid / id), 2)
    avg_price_low = round((sum_price_low / id), 2)
    ang_price_avg = round((sum_price_avg / id), 2)
    avg_sum_volume = round((sum_volume / id), 2)

    for add_date in add_date_list:  #在dataframe最後面補齊缺失的日期
        df.loc[id] = [add_date, '105 台北花市', 'FS443 香水百合 馬可波羅粉三朵', avg_highest_price, avg_price_high, avg_price_mid, avg_price_low, ang_price_avg, avg_sum_volume]
        id += 1
    df = df.sort_values(by=['日　　期'])    #再對日期欄做排序

    return df

org_path = r'D:\dataset\lilium_price\org\108\FS443.xls' #原始資料
path = r'D:\dataset\lilium_price\108\FS443.csv' #生成資料
df = pd.read_excel(org_path, header=4)

# x = df.loc[df['日　　期'] == '108/01/10'] #查詢特定日期

i = 0
predate = 0
lost_date_list = []
sum_highest_price = 0
sum_price_high = 0
sum_price_mid = 0
sum_price_low = 0
sum_price_avg = 0
sum_volume = 0

with tqdm(total=len(df)) as pbar:
    df = df.drop(['增減%', '增減%.1', '殘貨量', 'Unnamed: 12'], axis=1) #刪除特定欄位
    for index, row in df.iterrows():
        data_row = df.loc[index].values
        date, market, product, highest_price, price_high, price_mid, price_low, price_avg, volume= data_row    #日期、市場、產品、最高價、上價、中價、下價、平均價、交易量        
        if date == '小　　計':
            break
        #各欄加總
        sum_highest_price += highest_price
        sum_price_high += price_high
        sum_price_mid += price_mid
        sum_price_low += price_low
        sum_price_avg += price_avg
        sum_volume += volume

        month = date.split('/')[1]
        date = date.split('/')[2]
        date = (int(month)-1)*31 + int(date)
        if not date - predate == 1: #前後兩個日期不相臨
            for i in range( date - predate - 1):    #找出中間所有缺少的日期
                lost_date = predate + i + 1
                lost_date_list.append(lost_date)
        predate  = date       
        df.loc[index, '日　　期'] = date    #日期欄位改寫為 (月*31 + 日)

        # 更新進度條
        pbar.update(1)
        pbar.set_description('preprocessing')

df = fill_df(df, lost_date_list, sum_highest_price, sum_price_high, sum_price_mid, sum_price_low, sum_price_avg, sum_volume)
df.to_csv(path, encoding='utf_8_sig', index=False)