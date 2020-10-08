import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

input_shape = (1, 3, 3, 1)
input = layers.Input(shape=(3, 3, 1))
output = layers.Conv2D(1, kernel_size=3, strides=1, padding='same')(input)
print(output.shape)

model = models.Model(input, output)
model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
model.summary()

a = np.ones(shape=input_shape)
b = model.predict(a)
print(b)

conv_layer = model.layers[1]
print(conv_layer.get_weights())

for layer in model.layers:
    print(layer)