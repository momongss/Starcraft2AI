import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def small_action_model():
    entity_list = layers.Input(shape=(50, 46), name='entity_list')
    # game_loop = layers.Input(shape=(1,), name='game_loop')
    # player = layers.Input(shape=(11,), name='player')

    # action_type_logits = layers.LSTM(573, name='action_type_logits')(entity_list)
    lstm_out = layers.Bidirectional(layers.LSTM(573), name='lstm_out')(entity_list)

    # Dense
    action_type_logits = layers.Dense(573, name='action_type_logits', activation='so')(lstm_out)

    action_type_model = models.Model(inputs=[entity_list], outputs=[action_type_logits],
                                     name='action_type_model')
    action_type_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.mean_squared_error)

    return action_type_model


if __name__ == '__main__':
    model = small_action_model()
    model.summary()