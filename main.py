import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import *
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
X = pd.read_csv('csv_files/Data_X.csv')
Y = pd.read_csv('csv_files/Data_Y.csv')
df = pd.merge(X, Y, on='ID', how='inner')

# fill null values
df.fillna(0, inplace=True)

# replace non numeric data
df.replace(to_replace=["FR", "DE"], value=[1, 0], inplace=True)

# normalize data
cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'COUNTRY', 'TARGET')]
print(df[cols].describe().transpose()[['mean', 'std', 'min', 'max']])
df[cols] = StandardScaler().fit_transform(df[cols])
print(df[cols].describe().transpose()[['mean', 'std', 'min', 'max']])

# drop highly correlated data
corr_mat = df[cols].corr()
for i in range(len(corr_mat)):
    for j in range(i):
        if abs(corr_mat.values[i][j]) > 0.8 and corr_mat.columns[i] in df.columns:
            df.drop(corr_mat.columns[i], axis=1, inplace=True)

# get relevant columns to analyze
input_cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'TARGET')]


def test_model(model, columns):
    y_predict = cross_val_predict(model, df[columns], df["TARGET"])
    y_test = df["TARGET"].values
    print("R2 score =", round(r2_score(y_test, y_predict), 2))
    print("spearman =", round(spearmanr(y_test, y_predict).correlation, 2))
    print("RMSE =", round(mean_squared_error(y_test, y_predict, squared=False), 2))


# get the best columns to perform a regression analysis
K_best = SelectKBest(mutual_info_regression, k=10)
K_best.fit_transform(df[input_cols].values, df["TARGET"])
reg_cols = df[input_cols].columns[K_best.get_support()]

# test the different models
test_model(LinearRegression(), reg_cols)
test_model(Ridge(45), reg_cols)
test_model(Lasso(100), input_cols)
test_model(KNeighborsRegressor(12), input_cols)
test_model(DecisionTreeRegressor(max_depth=9), input_cols)


# create predictions for the new data
model = LinearRegression()
model.fit(df[reg_cols], df["TARGET"])

df_test = pd.read_csv('csv_files/DataNew_X.csv')
df_test.fillna(0, inplace=True)
df_test.replace(to_replace=["FR", "DE"], value=[1, 0], inplace=True)

Y_test_submission = df_test[['ID']].copy()
Y_test_submission['TARGET'] = model.predict(df_test[reg_cols])

Y_test_submission.to_csv('DataNew_Y_pred.csv', index=False)
