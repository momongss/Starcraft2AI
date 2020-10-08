import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 2,955,264

# LSTM
# input : Scalar_encoder, Entity_encoder, Spatial_encoder
# output : LSTM output   : 1D 256

def Core():
    Scalar_encoder = layers.Input(shape=(128,), dtype='float32')
    Entity_encoder = layers.Input(shape=(128,), dtype='float32')
    Spatial_encoder = layers.Input(shape=(128,), dtype='float32')

    # LSTM_input = layers.concatenate([Scalar_encoder, Entity_encoder, Spatial_encoder])
    LSTM_input = layers.Input()
    print(LSTM_input.shape)
    x = layers.LSTM(384, input_shape=(384, 1))(LSTM_input)
    x = layers.LSTM(384)(x)
    lstm_out = layers.LSTM(384)(x)
    # LSTM_model = models.Model(inputs=[Scalar_encoder, Entity_encoder, Spatial_encoder], outputs=[lstm_out])
    LSTM_model = models.Model(inputs=LSTM_input, outputs=lstm_out)
    return LSTM_model

def core_sequential():
    input_size = 384
    hidden_size = 384
    model = models.Sequential()

    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))
    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))
    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))

    model.compile(optimizer='adam', loss=keras.losses.KLDivergence())
    return model

model = core_sequential()
model.summary()


# import numpy as np
#
# a = np.array([[[1], [2], [3]]])
# print(a.shape)
# print(model.predict(a))