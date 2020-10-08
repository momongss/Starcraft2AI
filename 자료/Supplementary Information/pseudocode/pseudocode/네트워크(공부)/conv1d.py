from tensorflow import keras
from tensorflow.keras import layers, models

inputs = keras.layers.Input(shape=(64, 64, 11), name='lstm_output')

layers.Conv2D(filters=100, kernel_size=4, strides=(1, 1), padding="valid")
outputs = layers.Conv2D(1, 3, padding='same')(inputs)

model = models.Model(inputs=inputs, outputs=outputs, name='toy_resnet')
model.summary()