# https://ychai.uk/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/

from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 9,508,412

# action type head
# fc 버전.

# width = 256, 10 layers => parameter : 723,712

def Action_type():
    layer_width = 1024
    output_size = 572

    lstm_output = keras.Input(shape=(256,), name='lstm_output')
    scalar_context = keras.Input(shape=(256,), name='scalar_context')
    action_type_inputs = keras.layers.concatenate([lstm_output, scalar_context])

    x = layers.Dense(layer_width, activation='relu')(action_type_inputs)
    x = layers.Dense(layer_width, activation='relu')(x)
    x = layers.Dense(layer_width, activation='relu')(x)
    x = layers.Dense(layer_width, activation='relu')(x)
    x = layers.Dense(layer_width, activation='relu')(x)

    x = layers.Dense(layer_width, activation='relu')(x)
    x = layers.Dense(layer_width, activation='relu')(x)
    x = layers.Dense(layer_width, activation='relu')(x)
    x = layers.Dense(layer_width, activation='relu')(x)

    action_type_logits = layers.Dense(output_size, activation='softmax')(x)

    model = models.Model(inputs=[lstm_output, scalar_context], outputs=action_type_logits, name='action_type')
    return model

model = Action_type()
model.summary()