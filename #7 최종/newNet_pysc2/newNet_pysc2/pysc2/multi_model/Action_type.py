import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# action type head
# resblock 버전.
# convolution layer의 width를 얼마로 해야하는지 정보가 없음. ??

# Total params: 5,969,212
# Trainable params: 5,944,636
# Non-trainable params: 24,576

# input : lstm_output, scalar_context
# output : action_type_logits


def action_type_encoder(lstm_output, scalar_context):
    MLP_LEN = 256
    action_type_len = 573

    # lstm_output = keras.layers.Input(shape=(384, ), name='lstm_output')
    # scalar_context = keras.Input(shape=(160,), name='scalar_context')

    Resblock_output = layers.Flatten()(lstm_output)
    Resblock_output = layers.BatchNormalization()(Resblock_output)
    # Resblock_output = layers.Activation('sigmoid')(Resblock_output)
    # scalar_context = layers.Activation('sigmoid')(scalar_context)

    Resblock_layer = 16

    for _ in range(Resblock_layer):
        x = layers.Dense(MLP_LEN)(Resblock_output)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(MLP_LEN)(x)
        x = layers.BatchNormalization(scale=False)(x)
        Resblock_output = layers.add([x, Resblock_output])
        Resblock_output = layers.Activation('relu')(Resblock_output)
        # Resblock_output = layers.Activation('sigmoid')(Resblock_output)

    Resblock_flatten = layers.Flatten()(Resblock_output)
    gate = layers.Dense(MLP_LEN, activation='sigmoid')(scalar_context)
    # layers.Activation.
    gate_input = layers.multiply([Resblock_flatten, gate])
    # action_type_logits = layers.Dense(action_type_len, activation='softmax', name='action_type_logits')(gate_input)
    x = layers.Dense(action_type_len)(gate_input)
    x = layers.BatchNormalization()(x)
    action_type_logits = layers.Activation('sigmoid', name='action_type_logits')(x)

    action_type = keras.backend.argmax(action_type_logits)
    action_type_one_hot = keras.backend.one_hot(action_type, action_type_len)

    gate_input = layers.Dense(MLP_LEN, activation='relu')(action_type_one_hot)
    gate = layers.Dense(MLP_LEN, activation='sigmoid')(scalar_context)
    gate_input = layers.multiply([gate_input, gate])
    autoregressive_embedding = layers.Dense(1024)(gate_input)

    gate = layers.Dense(MLP_LEN, activation='sigmoid')(scalar_context)
    gate_input = layers.multiply([lstm_output, gate])
    lstm_projection = layers.Dense(1024)(gate_input)

    autoregressive_embedding = layers.add([autoregressive_embedding, lstm_projection], name='autoregressive_embedding')

    action_type_encoder_model = models.Model(inputs=[lstm_output, scalar_context],
                                             outputs={'action_type_logits': action_type_logits,
                                                      'action_type': action_type,
                                                      'autoregressive_embedding': autoregressive_embedding},
                                             name='action_type(resnet)')
    return action_type_encoder_model


if __name__ == '__main__':
    model = action_type()
    model.summary()