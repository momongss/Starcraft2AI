import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# LSTM
entity_list = layers.Input(shape=(50, 46), name='entity_list')
x = layers.LSTM(573, return_sequences=True)(entity_list)
entity_out = layers.LSTM(573, name='entity_out')(x)

# atari net
minimap = layers.Input(shape=(128, 128, 11), name='minimap')
x = layers.Conv2D(filters=64, kernel_size=(8, 8), strides=(4, 4), padding='same', activation='relu')(minimap)
x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(x)
conv_out = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
conv_flatten = layers.Flatten()(conv_out)
minimap_out = layers.Dense(256, activation='relu', name='minimap_out')(conv_flatten)

concat = layers.concatenate([entity_out, minimap_out])

output = layers.Dense(573, activation='softmax')(concat)

# player = layers.Input(shape=(11,), name='player')
# dense

action_type_model = models.Model(inputs=[entity_list, minimap], outputs=output)
action_type_model.summary()