import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# LSTM
# input : Scalar_encoder, Entity_encoder, Spatial_encoder
# output : LSTM output   : 1D 256

Scalar_encoder = layers.Input(shape=(128,), dtype='float32')
Entity_encoder = layers.Input(shape=(128,), dtype='float32')
Spatial_encoder = layers.Input(shape=(128,), dtype='float32')

LSTM_input = layers.concatenate([Scalar_encoder, Entity_encoder, Spatial_encoder])
x = layers.Embedding(input_dim=1, output_dim=1, input_length=384)(LSTM_input)
# x = layers.LSTM(384)(x)
# x = layers.Embedding(input_dim=1, output_dim=1, input_length=384)(x)
# x = layers.LSTM(384)(x)
# x = layers.Embedding(input_dim=1, output_dim=1, input_length=384)(x)
lstm_out = layers.LSTM(384)(x)

LSTM_model = models.Model(inputs=[Scalar_encoder, Entity_encoder, Spatial_encoder], outputs=[lstm_out])
LSTM_model.summary()