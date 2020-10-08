from tensorflow import keras
from tensorflow.keras import layers, models


def mlp(input_size, output_size, layer, width):
    inputs = keras.Input(shape=(input_size,), name='action_type_input')

    x = layers.Dense(width, activation='relu')(inputs)
    for _ in range(layer-1):
        x = layers.Dense(width, activation='relu')(x)
    outputs = layers.Dense(output_size, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='action_type_output')
    return model


model = mlp(100, 50, 5, 10)
model.summary()
