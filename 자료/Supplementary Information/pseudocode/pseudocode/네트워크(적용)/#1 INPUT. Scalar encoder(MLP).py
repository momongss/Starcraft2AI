from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 76,992
# Trainable params: 76,992
# Non-trainable params: 0

# input : Scalar_features[features를 제외한 obs의 나머지 모두] ( ... )
# output : embedded_scalar (608), scalar_context(160)


def scalar_encoder():
    single_select = layers.Input(shape=(7,))
    multi_select = layers.Input(shape=(1400,))
    build_queue = layers.Input(shape=(140,))
    cargo = layers.Input(shape=(140,))
    production_queue = layers.Input(shape=(10,))
    last_actions = layers.Input(shape=(1,))
    cargo_slots_available = layers.Input(shape=(1,))
    home_race_requested = layers.Input(shape=(1,))
    away_race_requested = layers.Input(shape=(1,))
    action_result = layers.Input(shape=(1,))
    alerts = layers.Input(shape=(2,))
    game_loop = layers.Input(shape=(1,))
    score_cumulative = layers.Input(shape=(13,))
    score_by_category = layers.Input(shape=(55,))
    score_by_vital = layers.Input(shape=(9,))
    player = layers.Input(shape=(11,))
    control_groups = layers.Input(shape=(20,))
    upgrades = layers.Input(shape=(1,))
    available_actions_one_hot = layers.Input(shape=(573,))

    # action type 정하는데 필요한 것들 --> scalar_context

    single_select_out = layers.Dense(32, activation='relu')(single_select)
    multi_select_out = layers.Dense(32, activation='relu')(multi_select)
    build_queue_out = layers.Dense(32, activation='relu')(build_queue)
    cargo_out = layers.Dense(32, activation='relu')(cargo)
    production_queue_out = layers.Dense(32, activation='relu')(production_queue)
    last_actions_out = layers.Dense(32, activation='relu')(last_actions)
    cargo_slots_available_out = layers.Dense(32, activation='relu')(cargo_slots_available)
    home_race_requested_out = layers.Dense(32, activation='relu')(home_race_requested)
    away_race_requested_out = layers.Dense(32, activation='relu')(away_race_requested)
    action_result_out = layers.Dense(32, activation='relu')(action_result)
    alerts_out = layers.Dense(32, activation='relu')(alerts)
    game_loop_out = layers.Dense(32, activation='relu')(game_loop)
    score_cumulative_out = layers.Dense(32, activation='relu')(score_cumulative)
    score_by_category_out = layers.Dense(32, activation='relu')(score_by_category)
    score_by_vital_out = layers.Dense(32, activation='relu')(score_by_vital)
    player_out = layers.Dense(32, activation='relu')(player)
    control_groups_out = layers.Dense(32, activation='relu')(control_groups)
    upgrades_out = layers.Dense(32, activation='relu')(upgrades)
    available_actions_one_hot_out = layers.Dense(32, activation='relu')(available_actions_one_hot)

    scalar_context = layers.concatenate([
        away_race_requested_out,
        available_actions_one_hot_out,
        single_select_out,
        multi_select_out,
        control_groups_out
    ], name='scalar_context')

    embedded_scalar = layers.concatenate([single_select_out,
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
                       available_actions_one_hot_out], name='embedded_scalar')

    scalar_encoder_model = models.Model(inputs=[single_select,
                        multi_select,
                       build_queue,
                       cargo,
                       production_queue,
                       last_actions,
                       cargo_slots_available,
                       home_race_requested,
                       away_race_requested,
                       action_result,
                       alerts,
                       game_loop,
                       score_cumulative,
                       score_by_category,
                       score_by_vital,
                       player,
                       control_groups,
                       upgrades,
                       available_actions_one_hot], outputs=[embedded_scalar, scalar_context])

    return scalar_encoder_model


if __name__ == '__main__':
    scalar_model = scalar_encoder()
    scalar_model.summary()