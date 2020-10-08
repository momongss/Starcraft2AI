import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def get_action_Model():
    # Input
    minimap = layers.Input(shape=(128, 128, 11), name='feature_minimap')
    player = layers.Input(shape=(11,), name='player')
    game_loop = layers.Input(shape=(1,), name='game_loop')

    # game info
    info = layers.concatenate([player, game_loop])
    info_fc = layers.Dense(64, activation='tanh')(info)

    # map
    x = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')(minimap)
    minimap_out = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    minimap_flatten = layers.Flatten()(minimap_out)

    # action_type
    map_fc = layers.Dense(512, activation='relu')(minimap_flatten)
    feat_fc = layers.concatenate([map_fc, info_fc])
    x = layers.Dense(256, activation='relu')(feat_fc)
    x = layers.Dropout(0.2)(x)
    action_type_logits = layers.Dense(3, activation='softmax', name='action_type_logits')(x)

    action_model = models.Model(inputs=[minimap, player, game_loop],
                              outputs=[action_type_logits])

    action_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       # loss=keras.losses.CategoricalCrossentropy(),
                       loss=keras.losses.KLDivergence())

    return action_model


def get_point_Model():
    # Input
    minimap = layers.Input(shape=(128, 128, 11), name='feature_minimap')

    conv = layers.Conv2D(filters=16, kernel_size=8, strides=4, padding='same')(minimap)
    conv = layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(conv)
    conv_flatten = layers.Flatten()(conv)

    dense = layers.Dense(512, activation='relu')(conv_flatten)
    dense = layers.Dropout(0.2)(dense)
    point_logits = layers.Dense(7, activation='softmax', name='attack_point_logits')(dense)

    point_model = models.Model(inputs=[minimap], outputs=[point_logits])

    point_model.compile(optimizer=keras.optimizers.Adam(0.001),
                        loss=keras.losses.KLDivergence())

    return point_model


if __name__ == '__main__':
    # model = get_action_Model()
    model = get_point_Model()
    model.summary()