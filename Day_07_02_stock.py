# Day_07_02_stock.py
import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection


# 퀴즈
# stock_daily.csv 파일로부터 x, y를 반환하는 함수를 만드세요
# batch_size, seq_length, n_features = 32, 7, 5

# 퀴즈
# 80%의 데이터로 학습하고 20%의 데이터에 대해 결과를 예측하세요
def get_xy():
    stock = pd.read_csv('data/stock_daily.csv', skiprows=2, header=None)
    # print(stock)      # [732 rows x 5 columns]

    # values = preprocessing.scale(stock.values)
    values = preprocessing.minmax_scale(stock.values)
    values = values[::-1]

    grams = nltk.ngrams(values, 7+1)
    grams = np.float32(list(grams))
    # print(grams.shape)            # (725, 8, 5)

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])
    # print(x.shape, y.shape)       # (725, 7, 5) (725, 1)

    return x, y


# 퀴즈
# 앞에서 만든 데이터에 대해 모델을 구축하세요
def model_stock():
    x, y = get_xy()

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse,
                  metrics='mae')

    model.fit(x_train, y_train, epochs=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


model_stock()
