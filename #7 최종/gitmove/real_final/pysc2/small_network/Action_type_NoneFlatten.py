import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def small_action_model():
    entity_list = layers.Input(shape=(50, 46), name='entity_list')
    game_loop = layers.Input(shape=(1,), name='game_loop')
    player = layers.Input(shape=(11,), name='player')

    lstm_out = layers.LSTM(92)(entity_list)

    action_type_model = models.Model(inputs=[entity_list, game_loop, player], outputs=[lstm_out],
                                     name='action_type_model')
    action_type_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.mean_squared_error)

    return action_type_model


if __name__ == '__main__':
    model = small_action_model()
    model.summary()