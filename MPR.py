### Многослойная перцептронная регрессия ###

import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy

from keras.models import Sequential  # Два варианты моделей
from keras.layers import Dense  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2


# Загружаем выборку
def getData(df):
    df.index = pd.to_datetime(df.index)
    values = df.values  # Вытаскиваем значение из data frame
    data = []  # Создаём пустую базу

    # Проходим по всем строкам данных и преобразовываем время в удобный формат
    for v in values:
        # Разбиваем на значения, раделитель - ; # Отбрасываем два первых значения - в них даты
        # v[0] = datetime.datetime.strptime(v[0], '%Y-%m-%d %H:%M:%S').timestamp()
        data.append(v)  # Добавляем элемент в базу

    return data


# Получаем данные из файла
def getDataFromFile(fileName):
    df = pd.read_csv(fileName)  # Считываем файл с помощью pandas
    df = df.sort_values('Datetime')
    return getData(df)


data = getDataFromFile(r'C:\Users\nasty\PycharmProjects\OxaProject\PJME_hourly.csv')

d = data
print(len(d))  # Сколько есть записей
data = np.array(data)  # Превращаем в numpy массив

# Препроцессинг данных
flatX = data[:, 0]
flatY = data[:, 1].reshape(-1, 1)

cutoff_point = int(len(flatX) * 0.6)

trainFlatX = flatX[:cutoff_point]
testFlatX = flatX[cutoff_point + 1:]

trainFlatY = flatY[:cutoff_point, ]
testFlatY = flatY[cutoff_point + 1:, ]
scaler = StandardScaler()
trainFlatY = scaler.fit_transform(trainFlatY)
testFlatY = scaler.transform(testFlatY)
print(len(trainFlatY))
print(len(testFlatY))


# Метод окна
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


# При look_back = 1 предсказывает значение по предыдущему
look_back = 10

trainX, trainY = create_dataset(trainFlatY, look_back)
testX, testY = create_dataset(testFlatY, look_back)

model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='linear'))
model.add(Dense(8))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
history = model.fit(trainX, trainY, batch_size=100, epochs=10, validation_data=(testX, testY))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Предсказываем значение и сравниваем его, построив график
yhat = model.predict(testX)

# Преобразовываем данные в исходный формат для проверки RMSE

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

plt.plot(yhat_inverse[0:300], label='predict')
plt.plot(testY_inverse[0:300], label='true')
plt.legend()
plt.show()

# RMSE
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)
