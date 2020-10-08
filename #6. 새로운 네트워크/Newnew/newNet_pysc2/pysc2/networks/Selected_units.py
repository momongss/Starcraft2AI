from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np


def real_argmax(entity_logits, real_entity_count):
    entity_logits = entity_logits[:real_entity_count]
    return np.argmax(entity_logits)

# Total params: 763,047


def selected_units_encoder(autoregressive_embedding, action_type, entity_list):
    # autoregressive_embedding = keras.Input(shape=(1024,), name='autoregressive_embedding')
    # action_type = keras.Input(shape=(1,), name='action_type')
    # entity_list = keras.Input(shape=(500, 46), name='entity_list')

    enc = layers.Dense(1)(entity_list)
    enc = layers.Flatten()(enc)

    dec = layers.Dense(500)(autoregressive_embedding)

    add = layers.add([enc, dec])
    tanh = keras.activations.tanh(add)
    selected_unit_logits = layers.Dense(500, name='selected_unit_logits')(tanh)

    return selected_unit_logits


if __name__ == '__main__':
    model = selected_units()
    model.summary()
