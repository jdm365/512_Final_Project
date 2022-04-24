import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('original_data/train.csv')

train, test = train_test_split(df, test_size=.14)
train.set_index('id', inplace=True)
test.set_index('id', inplace=True)


train.to_csv('official_data/train.csv')
test.to_csv('official_data/test.csv')