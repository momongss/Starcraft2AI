from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 2,612,224


def queued():
    embedding_output_size = 500  # 대충임.

    input_size = embedding_output_size  # autoregressive_embedding 의 공통 크기?

    layer_width = 1024

    inputs = keras.Input(shape=(input_size,), name='Queued_input')

    x = layers.Dense(layer_width)(inputs)
    x = layers.Dense(layer_width, activation='relu')(x)
    delay_logits = layers.Dense(layer_width, activation='softmax')(x)

    model = models.Model(inputs, delay_logits, name='Queued_output')
    return model

model = queued()
model.summary()