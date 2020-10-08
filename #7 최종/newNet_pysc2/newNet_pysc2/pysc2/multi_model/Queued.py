from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 328,706
# Trainable params: 328,706
# Non-trainable params: 0


def queued_encoder(autoregressive_embedding):
    layer_width = 256

    # autoregressive_embedding = keras.Input(shape=(1024,), name='Queued_input')

    x = layers.Dense(layer_width, activation='relu')(autoregressive_embedding)
    x = layers.Dense(layer_width, activation='relu')(x)
    queued_logits = layers.Dense(2, activation='softmax', name='queued_logits')(x)
    queued = keras.backend.argmax(queued_logits)

    queued_encoder_model = models.Model(inputs=autoregressive_embedding,
                                        outputs={'queued_logits': queued_logits,
                                                 'queued': queued},
                                        name='Queued_output')
    return queued_encoder_model


if __name__ == '__main__':
    model = queued()
    model.summary()