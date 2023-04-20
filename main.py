import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
import numpy as np

X = pd.read_csv('csv_files/Data_X.csv')
Y = pd.read_csv('csv_files/Data_Y.csv')
df = pd.merge(X, Y, on='ID', how='inner')

print(df)

null_indexes = np.where(pd.isnull(df))
df.replace(to_replace=np.nan, value=0, inplace=True)
print(df.drop(['ID', 'DAY_ID', 'COUNTRY'], axis=1).corr())
