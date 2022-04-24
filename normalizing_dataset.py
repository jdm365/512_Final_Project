import pandas as pd
import numpy as np

def min_max_norm(df, col_name):
    sup = max(df[col_name])
    inf = min(df[col_name])
    norm_val = df[col_name].subtract(inf).divide(sup - inf)
    return norm_val

train_og = pd.read_csv('official_data/train.csv')
test_og = pd.read_csv('official_data/test.csv')

train_idxs = train_og.index.values.tolist()
test_idxs = test_og.index.values.tolist()

whole_df = pd.concat([train_og, test_og])
cols_to_norm = ['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
for col in cols_to_norm:
    whole_df[col] = min_max_norm(whole_df, col)

train_normed = whole_df[whole_df['id'].isin(train_idxs)]
test_normed = whole_df[whole_df['id'].isin(test_idxs)]

train_normed.set_index('id', inplace=True)
test_normed.set_index('id', inplace=True)

train_normed.to_csv('normalized_data/train.csv')
train_normed.to_csv('normalized_data/test.csv')