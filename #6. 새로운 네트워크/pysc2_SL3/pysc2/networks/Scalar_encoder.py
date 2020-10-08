import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 76,992
# Trainable params: 76,992
# Non-trainable params: 0

# input : Scalar_features[features를 제외한 obs의 나머지 모두] ( ... )
# output : embedded_scalar (608), scalar_context(160)


def scalar_encoder(single_select, multi_select, build_queue, cargo, production_queue, last_actions,
                   cargo_slots_available, home_race_requested, away_race_requested, action_result,
                   alerts, game_loop, score_cumulative, score_by_category, score_by_vital, player,
                   control_groups, upgrades, available_actions_one_hot):
    # action type 정하는데 필요한 것들 --> scalar_context

    single_select_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(single_select)
    multi_select_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(multi_select)
    build_queue_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(build_queue)
    cargo_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(cargo)
    production_queue_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(production_queue)
    last_actions_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(last_actions)
    cargo_slots_available_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(cargo_slots_available)
    home_race_requested_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(home_race_requested)
    away_race_requested_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(away_race_requested)
    action_result_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(action_result)
    alerts_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(alerts)
    game_loop_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(game_loop)
    score_cumulative_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(score_cumulative)
    score_by_category_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(score_by_category)
    score_by_vital_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(score_by_vital)
    player_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(player)
    control_groups_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(control_groups)
    upgrades_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(upgrades)
    available_actions_one_hot_out = tf.compat.v1.keras.layers.Dense(32, activation='relu')(available_actions_one_hot)

    scalar_context = tf.compat.v1.keras.layers.concatenate([
        away_race_requested_out,
        available_actions_one_hot_out,
        single_select_out,
        multi_select_out,
        control_groups_out
    ])

    embedded_scalar = tf.compat.v1.keras.layers.concatenate([single_select_out,
                       multi_select_out,
                       build_queue_out,
                       cargo_out,
                       production_queue_out,
                       last_actions_out,
                       cargo_slots_available_out,
                       home_race_requested_out,
                       away_race_requested_out,
                       action_result_out,
                       alerts_out,
                       game_loop_out,
                       score_cumulative_out,
                       score_by_category_out,
                       score_by_vital_out,
                       player_out,
                       control_groups_out,
                       upgrades_out,
                       available_actions_one_hot_out])

    embedded_scalar = tf.compat.v1.keras.layers.Flatten()(embedded_scalar)

    return embedded_scalar, scalar_context


if __name__ == '__main__':
    scalar_model = scalar_encoder()
    scalar_model.summary()