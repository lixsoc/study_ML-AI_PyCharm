import random
import numpy as np
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses



# 235+17=252
# x : 235+17
# y : 252
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
        q = q + '#' * (digits * 2 + 1 - len(q))

        t = str(a + b)
        t += '#' * (digits + 1 - len(t))

        questions.append(q)
        expected.append(t)

    return questions, expected


def make_onehot(funcs, chr2idx):
    batch_size, seq_length, n_features = len(funcs), len(funcs[0]), len(chr2idx)
    ohv = np.zeros([batch_size, seq_length, n_features])
    # print(funcs[batch_size-1][seq_length-1])
    # print(funcs[0][0])
    for i in range(batch_size):
        for j in range(seq_length):
            # funcs[i][j]
            k = chr2idx[funcs[i][j]]
            ohv[i, j, k] = 1

    # for i, f in enumerate(funcs):
    #     for j, c in enumerate(f):
    #         k = chr2idx[c]
    #         ohv[i, j, k] = 1

    return ohv


def model_addition(x, y, x_train, x_test, y_train, y_test):

    model = Sequential()
    model.add(layers.InputLayer(input_shape=x.shape[1:]))

    # bad
    # model.add(layers.SimpleRNN(128, return_sequences=True))
    # model.add(layers.Reshape([4, -1]))
    # good
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.RepeatVector(y.shape[1]))
    model.add(layers.LSTM(128, return_sequences=True))

    model.add(layers.Dense(y.shape[-1], activation='softmax'))
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(0.01),
        loss=losses.categorical_crossentropy,
        metrics='acc'
    )
    model.fit(x_train, y_train, epochs=30, verbose=2, validation_data=(x_test, y_test))

    model.save('Day_08_02_addition.h5')
    return model


questions, expected = make_data(size=50000, digits=3)

vocab = '0123456789+#'

chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}
# print(chr2idx)      #{'0': 0, '1': 1, '2': 2, ..., '9': 9, '+': 10, '#': 11}
# print(idx2chr)      #{0: '0', 1: '1', 2: '2', ..., 9: '9', 10: '+', 11: '#'}
#
# print(questions[:3])        #['638+755', '92+555', '245+866']
# print(expected[:3])         #['1393', '647#', '1111']

x = make_onehot(questions, chr2idx)
y = make_onehot(expected, chr2idx)
x_train, x_test, y_train, y_test = tts(x, y, random_state=2)

model1 = model_addition(x, y, x_train, x_test, y_train, y_test)
model2 = models.load_model('Day_08_02_addition.h5')

for _ in range(10):
    idx = random.randrange(len(x_test))

    q = x_test[idx][np.newaxis]
    a = y_test[idx][np.newaxis]
    p1 = model1.predict(q)
    p2 = model2.predict(q)

    q_arg = np.argmax(q[0], axis=1)
    a_arg = np.argmax(a[0], axis=1)
    p_arg1 = np.argmax(p1[0], axis=1)
    p_arg2 = np.argmax(p2[0], axis=1)

    q_dec = ''.join([idx2chr[n] for n in q_arg]).replace('#', '')
    a_dec = ''.join([idx2chr[n] for n in a_arg]).replace('#', '')
    p_dec1 = ''.join([idx2chr[n] for n in p_arg1]).replace('#', '')
    p_dec2 = ''.join([idx2chr[n] for n in p_arg2]).replace('#', '')

    print('문제 :', q_dec)
    print('정답 :', a_dec)
    print('예측 :', p_dec1)
    print('예측 :', p_dec2)
    print()