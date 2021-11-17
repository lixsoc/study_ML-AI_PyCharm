import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
from matplotlib import pyplot as plt


def get_xy():
    stock = pd.read_csv('data/GOOG.csv', index_col=0)
    # print(stock)
    values = [
        stock['Open'],
        stock['High'],
        stock['Low'],
        stock['Volume'],
        stock['Close']
    ]
    values = np.transpose(values)

    scaler = preprocessing.MinMaxScaler()
    values = scaler.fit_transform(values)

    grams = nltk.ngrams(values, 3)
    grams = np.float32(list(grams))

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])

    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


def model_stock():
    x, y, data_min, data_max = get_xy()

    data = model_selection.train_test_split(x, y, random_state=20)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.mse,
        metrics='mae'
    )

    model.fit(x_train, y_train, epochs=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test)

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r', label='target')
    plt.plot(p, 'g', label='predict')
    plt.legend()

    p = data_min + (data_max - data_min) * p
    y_test = data_min + (data_max - data_min) * y_test

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r')
    plt.plot(p, 'g')
    plt.ylim(1000, 3000)

    plt.show()


model_stock()