import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 3,973,632

# LSTM
# input : embedded_entity, embedded_spatial, embedded_scalar
# output : LSTM output (256)


def core(embedded_scalar, embedded_entity, embedded_spatial, prev_state):
    # embedded_scalar = layers.Input(shape=(608,), dtype='float32')
    # embedded_entity = layers.Input(shape=(256,), dtype='float32')
    # embedded_spatial = layers.Input(shape=(256,), dtype='float32')
    # prev_state : 768
    # total : 1888
    prev_state = tf.compat.v1.keras.layers.Flatten()(prev_state)
    LSTM_input = tf.compat.v1.keras.layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial, prev_state])
    LSTM_input = tf.compat.v1.keras.layers.Reshape((1, 1888))(LSTM_input)

    # x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = tf.compat.v1.keras.layers.LSTM(384, return_sequences=True, stateful=True)(LSTM_input)
    x = tf.compat.v1.keras.layers.LSTM(384, return_sequences=True, stateful=True)(x)
    lstm_out, state_h, state_c = tf.compat.v1.keras.layers.LSTM(384, return_state=True, stateful=True)(x)
    next_state = tf.compat.v1.keras.layers.concatenate([state_h, state_c])
    next_state = tf.compat.v1.keras.layers.Flatten(name='next_state')(next_state)

    return lstm_out, next_state


# if __name__ == '__main__':
#     model = core()
#     model.summary()



def core_sequential():
    input_size = 384
    hidden_size = 384
    model = models.Sequential()

    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))
    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))
    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))

    model.compile(optimizer='adam', loss=keras.losses.KLDivergence())
    return model




# import numpy as np
#
# a = np.array([[[1], [2], [3]]])
# print(a.shape)
# print(model.predict(a))