import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np


X = pd.read_csv('csv_files/Data_X.csv')
Y = pd.read_csv('csv_files/Data_Y.csv')
df = pd.merge(X, Y, on='ID', how='inner')

df.replace(to_replace=np.nan, value=0, inplace=True)

cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'COUNTRY')]

print(pd.concat([df[cols].mean(), df[cols].var(), df[cols].min(), df[cols].max()], axis=1).set_axis(['MEAN', 'VAR', 'MIN', 'MAX'], axis=1))
df[cols] = StandardScaler().fit_transform(df[cols])
print(pd.concat([df[cols].mean(), df[cols].var(), df[cols].min(), df[cols].max()], axis=1).set_axis(['MEAN', 'VAR', 'MIN', 'MAX'], axis=1))


corr_mat = df[cols].corr()
for i in range(len(corr_mat)):
    for j in range(i):
        if abs(corr_mat.values[i][j]) > 0.5 and corr_mat.columns[i] in df.columns:
            df.drop(corr_mat.columns[i], axis=1, inplace=True)


input_cols = [e for e in df.columns if e not in ('ID', 'DAY_ID', 'COUNTRY', 'TARGET')]
train, test = train_test_split(df, test_size=0.2, random_state=69)

model = LinearRegression()
model.fit(train[input_cols], train["TARGET"])

y_predict = model.predict(test[input_cols])
y_test = test["TARGET"].values

print("Mean absolute error =", round(metrics.mean_absolute_error(y_test, y_predict), 2))
print("Mean squared error =", round(metrics.mean_squared_error(y_test, y_predict), 2))
print("Median absolute error =", round(metrics.median_absolute_error(y_test, y_predict), 2))
print("Explain variance score =", round(metrics.explained_variance_score(y_test, y_predict), 2))
print("R2 score =", round(metrics.r2_score(y_test, y_predict), 2))

model = Ridge(10)
model.fit(train[input_cols], train["TARGET"])

y_predict = model.predict(test[input_cols])
y_test = test["TARGET"].values

print("Mean absolute error =", round(metrics.mean_absolute_error(y_test, y_predict), 2))
print("Mean squared error =", round(metrics.mean_squared_error(y_test, y_predict), 2))
print("Median absolute error =", round(metrics.median_absolute_error(y_test, y_predict), 2))
print("Explain variance score =", round(metrics.explained_variance_score(y_test, y_predict), 2))
print("R2 score =", round(metrics.r2_score(y_test, y_predict), 2))

model = Lasso(0.1)
model.fit(train[input_cols], train["TARGET"])

y_predict = model.predict(test[input_cols])
y_test = test["TARGET"].values

print("Mean absolute error =", round(metrics.mean_absolute_error(y_test, y_predict), 2))
print("Mean squared error =", round(metrics.mean_squared_error(y_test, y_predict), 2))
print("Median absolute error =", round(metrics.median_absolute_error(y_test, y_predict), 2))
print("Explain variance score =", round(metrics.explained_variance_score(y_test, y_predict), 2))
print("R2 score =", round(metrics.r2_score(y_test, y_predict), 2))


model = KNeighborsRegressor(100)
model.fit(train[input_cols], train["TARGET"])

y_predict = model.predict(test[input_cols])
y_test = test["TARGET"].values

print("Mean absolute error =", round(metrics.mean_absolute_error(y_test, y_predict), 2))
print("Mean squared error =", round(metrics.mean_squared_error(y_test, y_predict), 2))
print("Median absolute error =", round(metrics.median_absolute_error(y_test, y_predict), 2))
print("Explain variance score =", round(metrics.explained_variance_score(y_test, y_predict), 2))
print("R2 score =", round(metrics.r2_score(y_test, y_predict), 2))

model = DecisionTreeRegressor()
model.fit(train[input_cols], train["TARGET"])

y_predict = model.predict(test[input_cols])
y_test = test["TARGET"].values

print("Mean absolute error =", round(metrics.mean_absolute_error(y_test, y_predict), 2))
print("Mean squared error =", round(metrics.mean_squared_error(y_test, y_predict), 2))
print("Median absolute error =", round(metrics.median_absolute_error(y_test, y_predict), 2))
print("Explain variance score =", round(metrics.explained_variance_score(y_test, y_predict), 2))
print("R2 score =", round(metrics.r2_score(y_test, y_predict), 2))
