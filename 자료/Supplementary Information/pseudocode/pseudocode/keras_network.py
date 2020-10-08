import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# LSTM
# input : Scalar_encoder, Entity_encoder, Spatial_encoder
# output : LSTM output   : 1D 256

Scalar_encoder = tf.keras.layers.Input(shape=(100,), dtype='float32')
Entity_encoder = tf.keras.layers.Input(shape=(100,), dtype='float32')
Spatial_encoder = tf.keras.layers.Input(shape=(100,), dtype='float32')

LSTM_input = tf.keras.layers.concatenate([Scalar_encoder, Entity_encoder, Spatial_encoder])
x = tf.keras.layers.Embedding(input_dim=1000, output_dim=512, input_length=300, name="embedding")(LSTM_input)
lstm_out = tf.keras.layers.LSTM(32, name='LSTM')(x)

LSTM_model = tf.keras.models.Model(inputs=[Scalar_encoder, Entity_encoder, Spatial_encoder], outputs=[lstm_out])
LSTM_model.summary()

# Action type   (Residual)
LSTM_output = tf.keras.layers.Input(shape=(256,), dtype='float32')
inputs = keras.Input(shape=(32, 32, 3), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name='toy_resnet')
model.summary()

#
# auxiliary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#
# auxiliary_input = tf.keras.layers.Input(shape=(5,), name='aux_input')
# x = tf.keras.layers.concatenate([lstm_out, auxiliary_input])
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
#
# main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)
#
# model = tf.keras.models.Model(inputs=[LSTM_input, auxiliary_input], outputs=[main_output, auxiliary_output])
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.KLDivergence())
# model.summary()