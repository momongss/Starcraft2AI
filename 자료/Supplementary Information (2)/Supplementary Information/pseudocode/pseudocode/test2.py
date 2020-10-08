from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(256, 1), name='img')
x = layers.Conv1D(32, 3, activation='relu')(inputs)
x = layers.Conv1D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling1D(3)(x)

x = layers.Conv1D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv1D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv1D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name='toy_resnet')
model.summary()