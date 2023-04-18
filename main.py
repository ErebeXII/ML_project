import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn

X = pd.read_csv('csv_files/Data_X.csv')
Y = pd.read_csv('csv_files/Data_Y.csv')
df = pd.merge(X, Y, on='ID', how='inner')

print(df)
