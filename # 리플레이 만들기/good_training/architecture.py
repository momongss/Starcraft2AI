import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


def get_compiled_Model():
    # Input
    feature_screen = layers.Input(shape=(128, 128, 27), name='feature_screen', batch_size=1)
    minimap = layers.Input(shape=(128, 128, 11), name='feature_minimap', batch_size=1)
    player = layers.Input(shape=(11,), name='player', batch_size=1)
    game_loop = layers.Input(shape=(1,), name='game_loop', batch_size=1)

    info = layers.concatenate([player, game_loop])
    info_fc = layers.Dense(32, activation='tanh')(info)

    # screen
    x = layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), padding='same')(feature_screen)
    screen_out = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    screen_flatten = layers.Flatten()(screen_out)

    # map
    x = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same')(minimap)
    minimap_out = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    minimap_flatten = layers.Flatten()(minimap_out)

    feat_fc = layers.concatenate([screen_flatten, minimap_flatten, info_fc])
    feat_fc = layers.Dense(256, activation='relu')(feat_fc)

    screen_x1 = layers.Dense(128, activation='softmax', name='screen_x1')(feat_fc)
    screen_y1 = layers.Dense(128, activation='softmax', name='screen_y1')(feat_fc)
    screen_x2 = layers.Dense(128, activation='softmax', name='screen_x2')(feat_fc)
    screen_y2 = layers.Dense(128, activation='softmax', name='screen_y2')(feat_fc)

    minimap_x = layers.Dense(128, activation='softmax', name='minimap_x')(feat_fc)
    minimap_y = layers.Dense(128, activation='softmax', name='minimap_y')(feat_fc)

    # action_type
    action_type_logits = layers.Dense(10, activation='softmax', name='action_type_logits')(feat_fc)
    good_model = models.Model(inputs=[feature_screen, minimap, player, game_loop],
                              outputs=[action_type_logits, screen_x1, screen_y1,
                                       screen_x2, screen_y2, minimap_x, minimap_y])

    action_w = K.variable(1.0)
    screen_x1_w = K.variable(0.0)
    screen_y1_w = K.variable(0.0)
    screen_x2_w = K.variable(0.0)
    screen_y2_w = K.variable(0.0)

    good_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                       # loss=keras.losses.CategoricalCrossentropy(),
                       loss=keras.losses.KLDivergence(),
                       loss_weights={
                           'action_type_logits': action_w,
                           'screen_x1': screen_x1_w,
                           'screen_y1': screen_y1_w,
                           'screen_x2': screen_x2_w,
                           'screen_y2': screen_y2_w,
                       })

    return good_model


if __name__ == '__main__':
    model = get_compiled_Model()
    model.summary()