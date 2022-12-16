import pandas as pd
df = pd.read_csv('DATASETS/CSV-01-12/01-12/DrDoS_UDP.csv').sample(37500)
df_test = pd.read_csv('DATASETS/CSV-03-11/03-11/UDP.csv').sample(12500)
df_valid = pd.read_csv('DATASETS/CSV-03-11/03-11/UDP.csv').sample(12500)

df.to_csv('train.csv')
df_test.to_csv('test.csv')
df_valid.to_csv('valid.csv')