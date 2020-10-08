from tensorflow import keras
from tensorflow.keras import layers, models

input_size = 256

layer_width = 256

inputs = keras.Input(shape=(input_size,), name='action_type_input')

x = layers.Dense(layer_width, activation='relu')(inputs)
x = layers.Dense(layer_width, activation='relu')(x)
x = layers.Dense(layer_width, activation='relu')(x)
x = layers.Dense(layer_width, activation='relu')(x)
x = layers.Dense(layer_width, activation='relu')(x)

x = layers.Dense(layer_width, activation='relu')(x)
x = layers.Dense(layer_width, activation='relu')(x)
x = layers.Dense(layer_width, activation='relu')(x)
x = layers.Dense(layer_width, activation='relu')(x)

outputs = layers.Dense(layer_width, activation='softmax')(x)

model = models.Model(inputs, outputs, name='action_type_output')
model.summary()