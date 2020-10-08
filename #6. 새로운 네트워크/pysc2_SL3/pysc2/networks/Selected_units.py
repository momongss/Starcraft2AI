import tensorflow as tf
import numpy as np


def real_argmax(entity_logits, real_entity_count):
    entity_logits = entity_logits[:real_entity_count]
    return np.argmax(entity_logits)

# Total params: 763,047


def selected_units_encoder(autoregressive_embedding, action_type, entity_list):
    # autoregressive_embedding = keras.Input(shape=(1024,), name='autoregressive_embedding')
    # action_type = keras.Input(shape=(1,), name='action_type')
    # entity_list = keras.Input(shape=(500, 46), name='entity_list')

    enc = tf.compat.v1.keras.layers.Dense(1)(entity_list)
    enc = tf.compat.v1.keras.layers.Flatten()(enc)

    dec = tf.compat.v1.keras.layers.Dense(500)(autoregressive_embedding)

    add = tf.compat.v1.keras.layers.add([enc, dec])
    tanh = tf.compat.v1.keras.activations.tanh(add)
    selected_unit_logits = tf.compat.v1.keras.layers.Dense(500, name='selected_unit_logits')(tanh)

    return selected_unit_logits


if __name__ == '__main__':
    model = selected_units()
    model.summary()
