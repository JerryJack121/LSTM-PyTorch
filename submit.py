import pandas as pd
import numpy as np
import os

submit_df_path = r'.\results\test_flower_price.csv'
submit_df = pd.read_csv(submit_df_path)
f = 10
for index, row in submit_df.iterrows():
    flower = row['flower_no']
    df = pd.read_csv(os.path.join(r'.\results',flower+'_result.csv'))
    submit_df.iloc[index, 3:7] = df.loc[(index//f)].values
submit_df.to_csv(os.path.join(r'.\results', 'submit.csv'), index=None)