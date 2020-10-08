import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 3,973,632

# LSTM
# input : embedded_entity, embedded_spatial, embedded_scalar
# output : LSTM output (256)


def core():
    embedded_scalar = layers.Input(shape=(608,), dtype='float32')
    embedded_entity = layers.Input(shape=(256,), dtype='float32')
    embedded_spatial = layers.Input(shape=(256,), dtype='float32')
    # 1120

    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(x)
    x = layers.LSTM(384, return_sequences=True)(x)
    lstm_out = layers.LSTM(384)(x)
    # LSTM_model = models.Model(inputs=[Scalar_encoder, Entity_encoder, Spatial_encoder], outputs=[lstm_out])
    LSTM_model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial], outputs=lstm_out)
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

model = core()
model.summary()


# import numpy as np
#
# a = np.array([[[1], [2], [3]]])
# print(a.shape)
# print(model.predict(a))