from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn import preprocessing as ppc
from sklearn.model_selection import train_test_split as tts

import numpy as np
import pandas as pd

# 1. 데이터 파일 읽어오기
# 2. 데이터 분류
# 3. 데이터 전처리

# 1. 데이터 파일 읽어오기
abalone = pd.read_csv('data/abalone.data', header=None)
print(abalone)

# 2. 데이터 분류
x = abalone.values[:, 1:]
y = abalone.values[:, :1]

x = ppc.scale(np.float32(x))
y = ppc.scale(np.float32(y))

enc = ppc.LabelEncoder()
y = enc.fit_transform(y)

# 3. 데이터 전처리
tr_x, te_x, tr_y, te_y = tts(x,y, test_size=0.2, random_state=42)

# 4. 러닝
model = keras.Sequential()
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=10, verbose=2)
print(model.evaluate(x_test, y_test, verbose=0))
