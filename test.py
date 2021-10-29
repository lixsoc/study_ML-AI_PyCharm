# day_01_01_linearRegression
# 선형회귀

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import pandas as pd
import numpy


def linear_reg():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = models.Sequential()
    model.add(layers.Dense(1))
    sgd = optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mse')
    model.fit(x, y, epochs=10)

    p = model.predict(x)
    p = p.reshape(-1)
    e = p - y

    print(e)


def linear_reg_csv(file_path: str):
    csv_data = pd.read_csv(file_path, index_col=0)

    # print(csv_data.values)      # [[  4   2] ...]

    # test case 1
    # x = csv_data.values[:, :-1]
    # y = csv_data.values[:, :1]
    # print(x.shape, x.shape)     # (50, 1) (50, 1)

    # test case 2
    x = csv_data.values[:, 0]
    y = csv_data.values[:, 1]
    # print(x.shape, x.shape)     # (50,) (50,)

    model = models.Sequential()
    model.add(layers.Dense(1))
    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='mse')
    history = model.fit(x, y, epochs=100)

    p = model.predict([30, 50])
    p = p.reshape(-1)
    p1, p2 = p
    print(p1, p2)

    print(model.weights)

    plt.plot(history.epoch, history.history['loss'], '-o', label='loss')
    plt.legend()
    plt.xlim(left=0)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    fig2 = plt.figure()
    plt.plot(x, y, '-o', label='data')
    plt.plot([0, 30],)
    plt.legend()
    plt.xlim(left=0)
    plt.xlabel('x_data')
    plt.ylabel('y_data')

    plt.show()

    # p = p.reshape(-1)
    # e = p - y
    # print(e)



# linear_reg()
# linear_reg_cars()

file_path = "C:\basic_ml&NN\data\cars.csv"
linear_reg_csv(file_path)

