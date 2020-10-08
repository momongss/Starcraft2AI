from tensorflow import keras
from tensorflow.keras import layers, models

inputs = keras.layers.Input(shape=(100,), name='lstm_output')

outputs = layers.Conv1D(100, 3, padding='same')(inputs)

model = models.Model(inputs=inputs, outputs=outputs, name='toy_resnet')
model.summary()