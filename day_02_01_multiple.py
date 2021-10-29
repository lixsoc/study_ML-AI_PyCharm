# day_02_01_multipleRegression

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def multiple_reg():
    x = [
        [1, 2],
        [0, 2],
        [3, 0],
        [0, 4],
        [5, 0],
    ]
    y = [
        [1],
        [2],
        [3],
        [4],
        [5],
    ]

    model = models.Sequential()
    model.add(layers.Dense(1))
    sgd = optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mse')
    model.fit(x, y, epochs=10)

    p = model.predict(x)
    p = p.reshape(-1)
    e = p - y

    print(p)


def multiple_reg_boston():
    # https://wikidocs.net/49966
    # boston 집값데이터
    # 기본 testset = 0.2
    boston = keras.datasets.boston_housing.load_data()
    train, test = boston
    print(type(train), type(test))              # <class 'tuple'> <class 'tuple'>

    x_train, y_train = train
    x_test, y_test = test
    print(x_train.shape, x_test.shape)           # (404, 13) (102, 13)
    print(type(x_train), type(x_test))          # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

    print(y_train.shape, y_test.shape)           # (404,) (102,)
    print(type(y_train), type(y_test))          # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    print(y_train.shape, y_test.shape)          # (404, 1) (102, 1)

    # # 같은 형태의 값이 맞는지 확인
    # print(x_train)
    # print()
    # print(x_test)

    model = models.Sequential()
    model.add(layers.Dense(1))
    sgd = optimizers.SGD(lr=0.000001)
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=1000, verbose=2)

    # 재구성 하지 않았을때 mae
    p = model.predict(x_test)
    e = p - y_test
    mae = np.mean(np.absolute(e))

    # 재구성 했을때 mae
    p_r = p.reshape(-1)
    y_test_r = y_test.reshape(-1)
    e_r = p_r - y_test_r
    mae_r = np.mean(np.absolute(e))

    print(p)
    print(f'mae : {mae}, mae_r : {mae_r}')


multiple_reg_boston()
