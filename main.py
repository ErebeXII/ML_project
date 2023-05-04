import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import *
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA

# Load the data
X = pd.read_csv('csv_files/Data_X.csv')
Y = pd.read_csv('csv_files/Data_Y.csv')
df = pd.merge(X, Y, on='ID', how='inner')

# fill null values
df.fillna(0, inplace=True)

# normalize data
cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'COUNTRY', 'TARGET')]
print(df[cols].describe().transpose()[['mean', 'std', 'min', 'max']])
df[cols] = StandardScaler().fit_transform(df[cols])
print(df[cols].describe().transpose()[['mean', 'std', 'min', 'max']])

# replace non numeric data
df.replace(to_replace=["FR", "DE"], value=[0, 1], inplace=True)

# drop highly correlated data
""" not useful anymore for PCA
corr_mat = df[cols].corr()
for i in range(len(corr_mat)):
    for j in range(i):
        if abs(corr_mat.values[i][j]) > 0.8 and corr_mat.columns[i] in df.columns:
            df.drop(corr_mat.columns[i], axis=1, inplace=True)
"""

# get relevant columns to analyze
input_cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'TARGET')]

# get the best number of components for PCA
"""
lin_reg = LinearRegression()
y = []
for i in range(1, len(input_cols)):
    pca = PCA(n_components=i)
    X_train_pc = pca.fit_transform(df[input_cols])
    y_predict = cross_val_predict(lin_reg, X_train_pc, df["TARGET"])
    y.append(spearmanr(df["TARGET"].values, y_predict).correlation)
plt.plot(range(1, len(input_cols)), y, '-o')
plt.show()
"""

# transform the datas using PCA
pca = PCA(n_components=19)
new_df = pd.DataFrame(pca.fit_transform(df[input_cols]))
input_cols = new_df.columns
new_df["TARGET"] = df["TARGET"]
df = new_df

# get the best columns to perform a regression analysis
K_best = SelectKBest(f_regression, k=8)
K_best.fit_transform(df[input_cols].values, df["TARGET"])
reg_cols = df[input_cols].columns[K_best.get_support()]

# optimize the hyperparameter of the models
""" 
y = []
K_best = SelectKBest(f_regression, k=8)
K_best.fit_transform(df[input_cols].values, df["TARGET"])
reg_cols = df[input_cols].columns[K_best.get_support()]
for i in range(1, len(input_cols) + 1):
    model = DecisionTreeRegressor(max_depth=i)
    y_predict = cross_val_predict(model, df[input_cols], df["TARGET"])
    y_test = df["TARGET"].values
    y.append(spearmanr(y_test, y_predict).correlation)
    print(i)
plt.plot(range(1, len(input_cols) + 1), y, '-o')
plt.show()
"""


def test_model(model, columns):
    y_predict = cross_val_predict(model, df[columns], df["TARGET"])
    y_test = df["TARGET"].values
    print("R2 score =", round(r2_score(y_test, y_predict), 4))
    print("spearman =", round(spearmanr(y_test, y_predict).correlation, 4))
    print("RMSE =", round(mean_squared_error(y_test, y_predict, squared=False), 4))


# test the different models
print("Linear regression:")
test_model(LinearRegression(), reg_cols)
print("Ridge regression:")
test_model(Ridge(1.7), reg_cols)
print("Lasso regression:")
test_model(Lasso(857), input_cols)
print("K-neighbors regression:")
test_model(KNeighborsRegressor(72), reg_cols)
print("Decision tree regression:")
test_model(DecisionTreeRegressor(max_depth=12), input_cols)
print("Random forest regression:")
test_model(RandomForestRegressor(n_estimators=100), input_cols)

# get and clean the validation data
df_test = pd.read_csv('csv_files/DataNew_X.csv')
df_test.fillna(0, inplace=True)
cols = [e for e in df_test.columns if e not in ('ID', 'DAY_ID', 'COUNTRY')]
df_test[cols] = StandardScaler().fit_transform(df_test[cols])
df_test.replace(to_replace=["FR", "DE"], value=[0, 1], inplace=True)
input_cols = [e for e in df_test.columns if e not in ('ID', 'DAY_ID', 'TARGET')]
new_df_test = pd.DataFrame(pca.transform(df_test[input_cols]))
new_df_test["ID"] = df_test["ID"]
df_test = new_df_test
input_cols = [e for e in df_test.columns if e != 'ID']


# create prediction for the validation data
Y_test_submission = df_test[['ID']].copy()
test_model = LinearRegression()
test_model.fit(df[reg_cols], df["TARGET"])
Y_test_submission['TARGET'] = test_model.predict(df_test[reg_cols])

# export the prediction in a .csv file
Y_test_submission.to_csv('DataNew_Y_pred.csv', index=False)
