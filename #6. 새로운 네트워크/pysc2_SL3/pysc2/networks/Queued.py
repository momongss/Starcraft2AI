import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 328,706
# Trainable params: 328,706
# Non-trainable params: 0


def queued_encoder(autoregressive_embedding):
    layer_width = 256

    x = tf.compat.v1.keras.layers.Dense(layer_width, activation='relu')(autoregressive_embedding)
    x = tf.compat.v1.keras.layers.Dense(layer_width, activation='relu')(x)
    queued_logits = tf.compat.v1.keras.layers.Dense(2, activation='softmax', name='queued_logits')(x)
    queued = tf.compat.v1.keras.backend.argmax(queued_logits)

    return queued_logits, queued


if __name__ == '__main__':
    model = queued()
    model.summary()