import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
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
"""
df.replace(to_replace=["FR", "DE"], value=[0, 1], inplace=True)
"""

# drop highly correlated data
""" not useful anymore for PCA
corr_mat = df[cols].corr()
for i in range(len(corr_mat)):
    for j in range(i):
        if abs(corr_mat.values[i][j]) > 0.8 and corr_mat.columns[i] in df.columns:
            df.drop(corr_mat.columns[i], axis=1, inplace=True)
"""

# separate data according to country
dfs = {"DE": df.copy(), "FR": df.copy()}
dfs["DE"].query("COUNTRY == 'DE'", inplace=True)
dfs["DE"].drop("COUNTRY", axis=1, inplace=True)
dfs["FR"].query("COUNTRY == 'FR'", inplace=True)
dfs["FR"].drop("COUNTRY", axis=1, inplace=True)


# get relevant columns to analyze
cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'TARGET', 'COUNTRY')]

cv = KFold(n_splits=10, shuffle=True, random_state=random.randint(1, 100))
# get the best number of components for PCA
def maximize_pca(country):
    lin_reg = LinearRegression()
    y = []
    for i in range(1, len(cols)):
        dfs[country] = dfs[country].sample(frac=1)
        pca = PCA(n_components=i)
        X_train_pc = pca.fit_transform(dfs[country][cols])
        y_predict = cross_val_predict(lin_reg, X_train_pc, dfs[country]["TARGET"], cv=cv)
        y.append(spearmanr(dfs[country]["TARGET"].values, y_predict).correlation)
    plt.plot(range(1, len(cols)), y, '-o')
    plt.show()


# transform the datas using PCA
pca = {"FR": PCA(n_components=17), "DE": PCA(n_components=19)}
input_cols = {}
for key in dfs:
    new_df = pd.DataFrame(pca[key].fit_transform(dfs[key][cols]))
    input_cols[key] = new_df.columns
    new_df["TARGET"] = dfs[key]["TARGET"].values
    dfs[key] = new_df

# get the best columns to perform a regression analysis
K_best = {"FR": SelectKBest(f_regression, k=11), "DE": SelectKBest(f_regression, k=9)}
reg_cols = {}
for key in dfs:
    K_best[key].fit_transform(dfs[key][input_cols[key]].values, dfs[key]["TARGET"])
    reg_cols[key] = dfs[key][input_cols[key]].columns[K_best[key].get_support()]


# optimize the hyperparameter of the models
def optimize_hyperparameter(country):
    y = []
    for i in range(1, 100):
        model = Ridge(i)
        y_predict = cross_val_predict(model, dfs[country][input_cols[country]], dfs[country]["TARGET"], cv=cv)
        y_test = dfs[country]["TARGET"].values
        y.append(spearmanr(y_test, y_predict).correlation)
        print(i)
    plt.plot(range(1, 100), y, '-o')
    plt.show()


optimize_hyperparameter("FR")

def test_model(model_FR, columns_FR, model_DE, columns_DE):
    y_predict = cross_val_predict(model_FR, dfs["FR"][columns_FR], dfs["FR"]["TARGET"], cv=cv)
    y_test = dfs["FR"]["TARGET"].values
    y_predict = np.concatenate((y_predict, cross_val_predict(model_DE, dfs["DE"][columns_DE], dfs["DE"]["TARGET"], cv=cv))).data
    y_test = np.concatenate((y_test, dfs["DE"]["TARGET"].values)).data
    print("R2 score =", round(r2_score(y_test, y_predict), 4))
    print("spearman =", round(spearmanr(y_test, y_predict).correlation, 4))
    print("RMSE =", round(mean_squared_error(y_test, y_predict, squared=False), 4))


# test the different models
print("Linear regression:")
test_model(LinearRegression(), reg_cols["FR"], LinearRegression(), reg_cols["DE"])
print("Ridge regression:")
test_model(Ridge(71), reg_cols["FR"], Ridge(11), reg_cols["DE"])
print("Lasso regression:")
test_model(Lasso(857), input_cols["FR"], Lasso(857), input_cols["DE"])
print("K-neighbors regression:")
test_model(KNeighborsRegressor(108), reg_cols["FR"], KNeighborsRegressor(103), reg_cols["DE"])
print("Decision tree regression:")
test_model(DecisionTreeRegressor(min_impurity_decrease=0.001, max_leaf_nodes=30), input_cols["FR"],
           DecisionTreeRegressor(min_impurity_decrease=0.01, max_leaf_nodes=15), input_cols["DE"])
"""
print("Random forest regression:")
test_model(RandomForestRegressor(n_estimators=100), input_cols["FR"], RandomForestRegressor(n_estimators=100), input_cols["DE"])
"""



# get and clean the validation data
df_test = pd.read_csv('csv_files/DataNew_X.csv')
df_test.fillna(0, inplace=True)
cols = [e for e in df_test.columns if e not in ('ID', 'DAY_ID', 'COUNTRY')]
df_test[cols] = StandardScaler().fit_transform(df_test[cols])

dfs_test = {"DE": df_test.copy(), "FR": df_test.copy()}
dfs_test["DE"].query("COUNTRY == 'DE'", inplace=True)
dfs_test["DE"].drop("COUNTRY", axis=1, inplace=True)
dfs_test["FR"].query("COUNTRY == 'FR'", inplace=True)
dfs_test["FR"].drop("COUNTRY", axis=1, inplace=True)

df_test.replace(to_replace=["FR", "DE"], value=[0, 1], inplace=True)

input_cols = [e for e in df_test.columns if e not in ('ID', 'DAY_ID', 'TARGET', 'COUNTRY')]
for key in dfs_test:
    new_df_test = pd.DataFrame(pca[key].transform(dfs_test[key][input_cols]))
    new_df_test["ID"] = dfs_test[key]["ID"].values
    dfs_test[key] = new_df_test

# create prediction for the validation data
Y_test_submissions = {}
models = {"FR": KNeighborsRegressor(108), "DE": KNeighborsRegressor(103)}
for key in dfs_test:
    Y_test_submissions[key] = dfs_test[key][['ID']].copy()
    models[key].fit(dfs[key][reg_cols[key]], dfs[key]["TARGET"])
    Y_test_submissions[key]['TARGET'] = models[key].predict(dfs_test[key][reg_cols[key]])

Y_test_submission = pd.concat([Y_test_submissions[key] for key in Y_test_submissions])

Y_test_submission = Y_test_submission.set_index('ID')
Y_test_submission = Y_test_submission.reindex(index=df_test['ID'])
Y_test_submission = Y_test_submission.reset_index()


# export the prediction in a .csv file
Y_test_submission.to_csv('DataNew_Y_pred2.csv', index=False)

X = pd.read_csv('DataNew_Y_pred.csv')
Y = pd.read_csv('DataNew_Y_pred2.csv')

print("spearman =", round(spearmanr(X["TARGET"], Y["TARGET"]).correlation, 4))
