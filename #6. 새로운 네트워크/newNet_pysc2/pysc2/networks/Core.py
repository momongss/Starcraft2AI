import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 3,973,632

# LSTM
# input : embedded_entity, embedded_spatial, embedded_scalar
# output : LSTM output (256)


def core(embedded_scalar, embedded_entity, embedded_spatial):
    # embedded_scalar = layers.Input(shape=(608,), dtype='float32')
    # embedded_entity = layers.Input(shape=(256,), dtype='float32')
    # embedded_spatial = layers.Input(shape=(256,), dtype='float32')
    # total : 1120
    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    LSTM_input = layers.Reshape((1, 1120))(LSTM_input)

    # x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(x)
    lstm_out = layers.LSTM(384)(x)

    return lstm_out


# if __name__ == '__main__':
#     model = core()
#     model.summary()



def core_sequential():
    input_size = 384
    hidden_size = 384

    LSTM_input = keras.layers.Input(shape=(1,1120))

    x = layers.LSTM(384, return_sequences=True)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(x)
    lstm_out, h_state, c_state = layers.LSTM(384, return_state=True)(x)

    model = models.Model(inputs=LSTM_input, outputs=[lstm_out, h_state, c_state])
    model.compile(optimizer='adam', loss=keras.losses.KLDivergence())
    return model


if __name__ == "__main__":
    model = core_sequential()
    model.summary()


# import numpy as np
#
# a = np.array([[[1], [2], [3]]])
# print(a.shape)
# print(model.predict(a))