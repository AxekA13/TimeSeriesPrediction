import pandas as pd

import datetime
import xgboost as xgb
import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy
from keras.models import Sequential  # Два варианты моделей
from keras.layers import Dense, LSTM, Dropout, Bidirectional  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2

data = pd.read_csv(r'C:\Users\nasty\PycharmProjects\OxaProject\cleaned_costs.csv', index_col=0)
data.head()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data = data.drop(['EC2-Instances(USD)','-','S3(USD)','Gesamtkosten (USD)','Steuer(USD)'], axis=1)

print(data)

split_date = '2019-06-19'
train = data.loc[data['Date'] <= split_date].copy()
test = data.loc[data['Date'] > split_date].copy()


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear

    X = df[['dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


X_train, y_train = create_features(train, label='EC2-Andere(USD)')
X_test, y_test = create_features(test, label='EC2-Andere(USD)')

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False)  # Change verbose to True if you want to see it train

test['EC2-Andere(USD)_Prediction'] = reg.predict(X_test)
pjme_all = pd.concat([test, train], sort=False)

_ = pjme_all[['EC2-Andere(USD)','EC2-Andere(USD)_Prediction']].plot(figsize=(15, 5))
plt.show()