import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy

from keras.models import Sequential, Input  # Два варианты моделей
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed,Dropout  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2

"""

#### Более правильное раскусывание временных рядов ###
# Функция "раскусывания" данных для временных рядов
# data - данные
# xLen - размер фрейма, по которому предсказываем
# xChannels - лист, номера каналов, по которым делаем анализ
# yChannels - лист, номера каналов, которые предсказываем
# stepsForward - на сколько шагов предсказываем в будущее
# если 0 - то на 1 шаг, можно использовать только при одном канале, указанном в yChannels
# xNormalization - нормализация входных каналов, 0 - нормальное распределение, 1 - к отрезку [0;1]
# yNormalization - нормализация прогнозируемых каналов, 0 - нормальное распределение, 1 - к отрезку [0;1]
# returnFlatten - делать ли одномерный вектор на выходе для Dense сетей
# valLen - сколько примеров брать для проверочной выборки (количество для обучающей посчитается автоматиески)
# convertToDerivative - bool, преобразовывали ли входные сигналы в производную
def getXTrainFromTimeSeries(data, xLen, xChannels, yChannels, stepsForward, xNormalization, yNormalization,
                            returnFlatten, valLen, convertToDerivative):
    # Если указано превращение данных в производную
    # То вычитаем поточечно из текущей точки предыдущую
    if (convertToDerivative):
        data = np.array([(d[1:] - d[:-1]) for d in data.T]).copy().T

    # Выбираем тип нормализации x
    # 0 - нормальное распределение
    # 1 - нормирование до отрезка 0-1
    if (xNormalization == 0):
        xScaler = StandardScaler()
    else:
        xScaler = MinMaxScaler()

    # Берём только те каналы, которые указаны в аргументе функции
    xData = data[:, xChannels]
    # Обучаем нормировщик
    xScaler.fit(xData)
    # Нормируем данные
    xData = xScaler.transform(xData)

    # Выбираем тип нормализации y
    # 0 - нормальное распределение
    # 1 - нормирование до отрезка 0-1
    if (yNormalization == 0):
        yScaler = StandardScaler()
    else:
        yScaler = MinMaxScaler()

    # Берём только те каналы, которые указаны в аргументе функции
    yData = data[:, yChannels]
    # Обучаем нормировщик
    yScaler.fit(yData)
    # Нормируем данные
    yData = yScaler.transform(yData)

    # Формируем xTrain
    # Раскусываем исходный ряд на куски xLen с шагом в 1
    xTrain = np.array([xData[i:i + xLen, xChannels] for i in range(xData.shape[0] - xLen - 1 - stepsForward)])

    # Формируем yTrain
    # Берём stepsForward шагов после завершения текущего x
    if (stepsForward > 0):
        yTrain = np.array([yData[i + xLen:i + xLen + stepsForward, yChannels] for i in
                           range(yData.shape[0] - xLen - 1 - stepsForward)])
    else:
        yTrain = np.array(
            [yData[i + xLen + stepsForward, yChannels] for i in range(yData.shape[0] - xLen - 1 - stepsForward)])

    # Делаем reshape y в зависимости от того
    # Прогнозируем на 1 шаг вперёд или на несколько
    if (stepsForward == 0):
        if ((len(yChannels) == 1)):
            yTrain = yTrain.reshape(yTrain.shape[0], 1)
    else:
        yTrain = yTrain.reshape(yTrain.shape[0], stepsForward)

    # Расчитыываем отступ между обучающими о проверочными данными
    # Чтобы они не смешивались
    xTrainLen = xTrain.shape[0]
    bias = xLen + stepsForward + 2

    # Берём из конечной части xTrain проверочную выборку
    xVal = xTrain[xTrainLen - valLen:]
    yVal = yTrain[xTrainLen - valLen:]
    # Оставшуюся часть используем под обучающую выборку
    xTrain = xTrain[:xTrainLen - valLen - bias]
    yTrain = yTrain[:xTrainLen - valLen - bias]

    # Если в функцию передали вернуть flatten сигнал (для Dense сети)
    # xTrain и xVal превращаем в flatten
    if (returnFlatten > 0):
        xTrain = np.array([x.flatten() for x in xTrain])
        xVal = np.array([x.flatten() for x in xVal])

    return (xTrain, yTrain), (xVal, yVal), (xScaler, yScaler)


# Формируем параметры загрузки данных
xLen = 1  # Анализируем по 300 прошедшим точкам
stepsForward = 0  # Предсказываем на 1 шаг вперёд
xChannels = list(range(data.shape[1]))  # Используем все входные каналы
yChannels = [0, 1, 2, 3, 4]  # Предказываем только open канал
xNormalization = 0  # Нормируем входные каналы стандартным распределением
yNormalization = 0  # Нормируем выходные каналы стандартным распределением
valLen = 30000  # Используем 30.000 записей для проверки
returnFlatten = 1  # Вернуть одномерные вечеторы
convertToDerivative = 0  # Не превращать в производную

# Загружаем данные
(xTrain, yTrain), (xVal, yVal), (xScaler, yScaler) = getXTrainFromTimeSeries(data, xLen, xChannels, yChannels,
                                                                             stepsForward, xNormalization,
                                                                             yNormalization, returnFlatten, valLen,
                                                                             convertToDerivative)


def getPred(currModel, xVal, yVal, yScaler):
    # Предсказываем ответ сети по проверочной выборке
    # И возвращаем исходны масштаб данных, до нормализации
    predVal = yScaler.inverse_transform(currModel.predict(xVal))
    yValUnscaled = yScaler.inverse_transform(yVal)

    return (predVal, yValUnscaled)


# Функция визуализирует графики, что предсказала сеть и какие были правильные ответы
# start - точка с которой начинаем отрисовку графика
# step - длина графика, которую отрисовываем
# channel - какой канал отрисовываем
def showPredict(start, step, channel, predVal, yValUnscaled):
    plt.plot(predVal[start:start + step, channel],
             label='Прогноз')
    plt.plot(yValUnscaled[start:start + step, channel],
             label='Базовый ряд')
    plt.xlabel('Время')
    plt.ylabel('Значение Close')
    plt.legend()
    plt.show()

"""
# Выводим параметры файла
data = pd.read_csv(r'C:\Users\nasty\PycharmProjects\OxaProject\cleaned_costs.csv', index_col=0)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date')
data = data.fillna(method='ffill')
print(data)
data = data.drop(['Steuer(USD)', 'Date'], axis=1)

scaler = StandardScaler()
data = data.values
print(data.shape)

cutoff_point = int(len(data[:, 0]) * 0.8)
train_data = data[:cutoff_point, ]
test_data = data[cutoff_point + 1:, ]
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


lookback = 3  # если =1, то предсказываем по предыдущему значению
count_of_predict = 2  # количество предсказываемых значений
xTrain, yTrain = split_sequence(train_data, lookback, count_of_predict)
xTest, yTest = split_sequence(test_data, lookback, count_of_predict)

features = 5  # Количество выходных слоёв

### Encoder-Decoder LSTM ###


model = Sequential()
model.add(LSTM(700, input_shape=(lookback, features), activation='sigmoid'))
model.add(RepeatVector(count_of_predict))
model.add(LSTM(700, return_sequences=True, activation='sigmoid'))
model.add(TimeDistributed(Dense(features)))
model.compile(loss='mse', optimizer='adam')
history = model.fit(xTrain, yTrain, epochs=200, validation_data=(xTest, yTest))

plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
plt.ylabel('Средняя ошибка')
plt.legend()
plt.show()
yhat = model.predict(xTest)

plt.plot(yhat[:, :, 1], label='predict')
plt.plot(yTest[:, :, 1], label='true')
plt.legend()
plt.show()
