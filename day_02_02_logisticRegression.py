# day_02_02_logisticRegression

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import sklearn.preprocessing as ppc


def logistic_regression():

    # 값의 예시 : 공부시간 및 출석
    x = [
        [1, 2],     #0
        [2, 1],
        [4, 5],     #1
        [5, 4],
        [8, 9],
        [9, 8],
    ]
    y = [
        [0],
        [0],
        [1],
        [1],
        [1],
        [1],
    ]

    model = models.Sequential()
    # model.add(layers.Dense(1, activation=keras.activation.sigmoid))
    model.add(layers.Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.01)
    # 2진 log 손실함수, acc = 정확도
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics='acc')
    # verbos : 나오는 ui생략
    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    # p = p.reshape(-1)
    # e = p - y
    a = np.int32(np.int32((p > 0.5)) == y)
    print(a)
    a = np.mean(a)
    print(a)


def logistic_regression_pima():
    # 스케일링이 되지 않은 데이터는 값이 제대로 안나옴!!!!!!

    # skiprows 으로 행삭제 가능
    # header = none 으로 헤더 리딩 안함.
    pima_data = pd.read_csv('data\pima-indians-diabetes.csv')
    # same code
    # pima_data_x = pima_data.values[:, :-1]
    pima_data_x = pima_data.values[:, 0:8]
    # !!!!!!!!!!!! 스케일링 !!!!!!!!!!!!!!!
    pima_data_x = ppc.scale(pima_data_x)
    # pima_data_x = ppc.minmax_scale(pima_data_x)
    # print(pima_data_x)

    # same code
    # pima_data_x = pima_data.values[:, -1:]
    pima_data_y = pima_data.values[:, 8:9]
    # print(pima_data_y)

    # sim code
    # train_size = int(len(x) * 0.7)
    # x_train, x_test = pima_data_x[:train_size], pima_data_x[train_size]
    x_train, x_test, y_train, y_test = tts(pima_data_x, pima_data_y, test_size=0.3, random_state=42)

    model = models.Sequential()
    # model.add(layers.Dense(1, activation=keras.activation.sigmoid))
    model.add(layers.Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.0001)
    # 2진 log 손실함수, acc = 정확도
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics='acc')
    # verbos : 나오는 ui생략
    history = model.fit(x_train, y_train, batch_size=50,
              epochs=1800,
              verbose=2,
              validation_data=(x_test, y_test))

    # p = model.predict(x_test)
    # print(p)
    # p = p.reshape(-1)
    # e = p - y
    # a = np.int32(np.int32((p > 0.5)) == y)
    # print(a)
    # a = np.mean(a)
    # print(a)

    plt.plot(history.epoch, history.history['acc'], '-o', label='acc')
    plt.plot(history.epoch, history.history['val_acc'], '-o', label='v_acc')
    plt.legend()
    plt.xlim(left=0)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()


logistic_regression_pima()
