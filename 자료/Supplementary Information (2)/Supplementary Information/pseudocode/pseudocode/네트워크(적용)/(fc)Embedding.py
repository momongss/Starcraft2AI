from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 182,282


def Embedding(action_onehot_size, scalar_context_size):
    output_size = 10

    action_type_onehot = keras.Input(shape=(action_onehot_size,), name='action_type_onehot')
    layer1 = layers.Dense(256, activation='relu', name='action_layer')(action_type_onehot)

    scalar_context = keras.Input(shape=(scalar_context_size,), name='scalar_context')

    gate = layers.Dense(256, activation='sigmoid', name='scalar_gate')(scalar_context)
    gated_input = layers.multiply([gate, layer1])
    autoregressive_embedding = layers.Dense(output_size)(gated_input)

    model = models.Model(inputs=[action_type_onehot, scalar_context], outputs=[autoregressive_embedding],
                         name='action_type_output')
    model.summary()
    return model


model = Embedding(action_onehot_size=572, scalar_context_size=128)
model.summary()