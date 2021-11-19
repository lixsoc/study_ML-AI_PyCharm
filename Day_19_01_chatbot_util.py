# Day_19_01_chatbot_util.py
import tensorflow.keras as keras
import numpy as np


def train_and_save_model(enc_x, dec_x, dec_y, vocab):
    # 인코더
    enc_input = keras.layers.Input(enc_x.shape[1:])
    _, enc_state = keras.layers.SimpleRNN(128, return_state=True)(enc_input)

    # 디코더
    dec_input = keras.layers.Input(dec_x.shape[1:])
    dec_output = keras.layers.SimpleRNN(128, return_sequences=True)(dec_input, initial_state=enc_state)
    dec_output = keras.layers.Dense(len(vocab), activation='softmax')(dec_output)

    model = keras.Model([enc_input, dec_input], dec_output)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit([enc_x, dec_x], dec_y, epochs=100, verbose=2)

    model.save('model/chat.h5')

