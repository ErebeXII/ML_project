import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn

datax = pd.read_csv('csv_files/Data_X.csv')
datay = pd.read_csv('csv_files/Data_Y.csv')
df = pd.merge(datax, datay, on='ID', how='inner')

print(df)