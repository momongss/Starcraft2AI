from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 182,282


def Embedding():
    output_size = 10

    autoregressive_embedding = keras.Input(shape=(1024,), name='autoregressive_embedding')
    layer1 = layers.Dense(256, activation='relu', name='action_layer')(autoregressive_embedding)

    scalar_context = keras.Input(shape=(160,), name='scalar_context')

    gate = layers.Dense(256, activation='sigmoid', name='scalar_gate')(scalar_context)
    gated_input = layers.multiply([gate, layer1])
    autoregressive_embedding = layers.Dense(output_size)(gated_input)

    model = models.Model(inputs=[autoregressive_embedding, scalar_context], outputs=autoregressive_embedding, name='action_type_output')
    model.summary()
    return model


model = Embedding()
model.summary()