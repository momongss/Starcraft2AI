from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 463,631
# Trainable params: 463,631
# Non-trainable params: 0


# (4,), (5,), (10,)
def target_unit_encoder(entity_embeddings, lstm_output):
    layer_width = 256

    # entity_embeddings = keras.Input(shape=(256,), name='entity_embeddings')
    # lstm_out = keras.Input(shape=(384,), name='Queued_input')

    shifted_input = layers.concatenate([entity_embeddings, lstm_output])
    x = layers.Dense(layer_width, activation='relu')(shifted_input)
    x = layers.Dense(layer_width, activation='relu')(x)
    shifted_logits4 = layers.Dense(4, activation='softmax', name='shifted_logits4')(x)

    x = layers.Dense(layer_width, activation='relu')(shifted_input)
    x = layers.Dense(layer_width, activation='relu')(x)
    shifted_logits5 = layers.Dense(5, activation='softmax', name='shifted_logits5')(x)

    x = layers.Dense(layer_width, activation='relu')(shifted_input)
    x = layers.Dense(layer_width, activation='relu')(x)
    control_group_id = layers.Dense(10, activation='softmax', name='control_group_id')(x)

    target_unit_encoder_model = models.Model(inputs=[entity_embeddings,
                                                     lstm_output],
                                             outputs={'shifted_logits4': shifted_logits4,
                                                      'shifted_logits5': shifted_logits5,
                                                      'control_group_id': control_group_id},
                                             name='target_unit')
    return target_unit_encoder_model


if __name__ == 'main':
    model = target_unit()
    model.summary()