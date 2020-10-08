import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def small_action_model():
    entity_list = layers.Input(shape=(50, 46), name='entity_list')
    game_loop = layers.Input(shape=(1,), name='game_loop')
    player = layers.Input(shape=(11,), name='player')

    entity_flatten = layers.Flatten()(entity_list)

    mlp_out1 = layers.Dense(1024, activation='relu')(entity_flatten)
    mlp_out1 = layers.Dropout(0.2)(mlp_out1)
    mlp_out1 = layers.Dense(1024, activation='relu')(mlp_out1)

    mlp_out1 = layers.concatenate([mlp_out1, player, game_loop])

    mlp_out2 = layers.Dense(768, activation='relu')(mlp_out1)
    Action_type_logits = layers.Dense(573, activation='softmax', name='action_type_logits')(mlp_out2)

    action_type_model = models.Model(inputs=[entity_list, game_loop, player], outputs=[Action_type_logits],
                                     name='action_type_model')
    action_type_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.mean_squared_error)

    return action_type_model


if __name__ == '__main__':
    model = small_action_model()
    model.summary()