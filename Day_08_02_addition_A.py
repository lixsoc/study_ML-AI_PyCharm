# Day_08_02_addition.py
import random
import tensorflow.keras as keras
from sklearn import model_selection

# 235+17=252
# x : 235+17
# y : 252


# 퀴즈
# 자릿수에 맞는 숫자를 만드는 함수를 구현하세요
# digits: 자릿수
import numpy as np


def make_number(digits):
    # return random.randrange(10 ** digits)

    d = random.randrange(digits) + 1
    return random.randrange(10 ** d)


def make_data(size, digits):
    questions, expected, seen = [], [], set()

    while len(questions) < size:
        a = make_number(digits)
        b = make_number(digits)

        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)

        q = '{}+{}'.format(a, b)
        q += '#' * (digits * 2 + 1 - len(q))    # 86+7###

        t = str(a + b)
        t += '#' * (digits + 1 - len(t))        # 93##

        questions.append(q)
        expected.append(t)

    return questions, expected


def make_onehot(texts, chr2idx):
    batch_size, seq_length, n_features = len(texts), len(texts[0]), len(chr2idx)
    v = np.zeros([batch_size, seq_length, n_features])

    for i, t in enumerate(texts):
        for j, c in enumerate(t):
            k = chr2idx[c]
            v[i, j, k] = 1
    return v


questions, expected = make_data(size=50000, digits=3)

vocab = '#+0123456789'

chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}
# print(chr2idx)        # {'#': 0, '+': 1, '0': 2, ...}
# print(idx2chr)        # {0: '#', 1: '+', 2: '0', ...}

# print(questions[:3])  # ['936+0##', '723+26#', '7+91###']
# print(expected[:3])   # ['936#', '749#', '98##']

x = make_onehot(questions, chr2idx)     # (50000, 7, 12)
y = make_onehot(expected, chr2idx)      # (50000, 4, 12)

# print(x[0, 0])    # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# print(x[0, -1])   # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 퀴즈
# 앞에서 만든 x, y에 대해 80%로 학습하고 20%에 대해 정확도를 구하세요
data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
# bad
# model.add(keras.layers.SimpleRNN(128, return_sequences=True))
# model.add(keras.layers.Reshape([4, -1]))
# excellent
model.add(keras.layers.SimpleRNN(128, return_sequences=False))
model.add(keras.layers.RepeatVector(y.shape[1]))                    # 3+1
model.add(keras.layers.SimpleRNN(128, return_sequences=True))
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=30, verbose=2,
          validation_data=(x_test, y_test))
# print(model.evaluate(x_test, y_test, verbose=0))






