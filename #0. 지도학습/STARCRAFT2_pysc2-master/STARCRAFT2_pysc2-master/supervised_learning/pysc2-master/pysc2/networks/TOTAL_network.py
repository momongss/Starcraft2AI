from pysc2.networks.Action_type import action_type_encoder
from pysc2.networks.Core import core
from pysc2.networks.Entitiy_encoder import entity_encoder
from pysc2.networks.Queued import queued_encoder
from pysc2.networks.Scalar_encoder import scalar_encoder
from pysc2.networks.Selected_units import selected_units_encoder
from pysc2.networks.Spatial_encoder import spatial_encoder
from pysc2.networks.Target_unit import target_unit_encoder
from pysc2.networks.Target_point import target_point_encoder

from tensorflow import keras
from tensorflow.keras import layers, models


def total_model():
    entity_list = layers.Input(shape=(500, 46), name='entity_list')
    minimap = layers.Input(shape=(128, 128, 11), name='minimap')

    # scalar features
    single_select = layers.Input(shape=(7,), name='single_select')
    multi_select = layers.Input(shape=(1400,), name='multi_select')
    build_queue = layers.Input(shape=(140,), name='build_queue')
    cargo = layers.Input(shape=(140,), name='cargo')
    production_queue = layers.Input(shape=(10,), name='production_queue')
    last_actions = layers.Input(shape=(1,), name='last_actions')
    cargo_slots_available = layers.Input(shape=(1,), name='cargo_slots_available')
    home_race_requested = layers.Input(shape=(1,), name='home_race_requested')
    away_race_requested = layers.Input(shape=(1,), name='away_race_requested')
    action_result = layers.Input(shape=(1,), name='action_result')
    alerts = layers.Input(shape=(2,), name='alerts')
    game_loop = layers.Input(shape=(1,), name='game_loop')
    score_cumulative = layers.Input(shape=(13,), name='score_cumulative')
    score_by_category = layers.Input(shape=(55,), name='score_by_category')
    score_by_vital = layers.Input(shape=(9,), name='score_by_vital')
    player = layers.Input(shape=(11,), name='player')
    control_groups = layers.Input(shape=(20,), name='control_groups')
    upgrades = layers.Input(shape=(1,), name='upgrades')
    available_actions_one_hot = layers.Input(shape=(573,), name='available_actions_one_hot')

    embedded_entity, entity_embeddings = entity_encoder(entity_list)
    map_skip, embedded_spatial = spatial_encoder(minimap)
    embedded_scalar, scalar_context = scalar_encoder(single_select, multi_select, build_queue, cargo, production_queue, last_actions,
                   cargo_slots_available, home_race_requested, away_race_requested, action_result,
                   alerts, game_loop, score_cumulative, score_by_category, score_by_vital, player,
                   control_groups, upgrades, available_actions_one_hot)

    lstm_output = core(embedded_scalar, embedded_entity, embedded_spatial)

    action_type_logits, action_type, autoregressive_embedding = action_type_encoder(lstm_output, scalar_context)

    queued_logits, queued = queued_encoder(autoregressive_embedding)

    selected_unit_logits = selected_units_encoder(autoregressive_embedding, action_type, entity_list)

    shifted_logits4, shifted_logits5, control_group_id = target_unit_encoder(entity_embeddings, lstm_output)

    minimap_logits, screen1_logits, screen2_logits = target_point_encoder(autoregressive_embedding, action_type, map_skip)

    starcraft_model = models.Model(inputs=[entity_list, minimap,
                single_select, multi_select, build_queue, cargo, production_queue, last_actions,
                cargo_slots_available, home_race_requested, away_race_requested, action_result,
                alerts, game_loop, score_cumulative, score_by_category, score_by_vital, player,
                control_groups, upgrades, available_actions_one_hot],
                outputs=[action_type_logits, queued_logits
                         ,selected_unit_logits, shifted_logits4, shifted_logits5, control_group_id
                         ,minimap_logits, screen1_logits, screen2_logits
                            # , screen1_logits, screen2_logits
                         ])

    starcraft_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.KLDivergence())
    return starcraft_model


if __name__ == '__main__':
    import numpy as np

    a = np.ones((1, 500, 46))
    b = np.ones((1, 128, 128, 11))

    single_select = np.ones((1, 7))
    multi_select = np.ones((1, 1400))
    build_queue = np.ones((1, 140))
    cargo = np.ones((1, 140))
    production_queue = np.ones((1, 10))
    last_actions = np.ones((1, 1))
    cargo_slots_available = np.ones((1, 1))
    home_race_requested = np.ones((1, 1))
    away_race_requested = np.ones((1, 1))
    action_result = np.ones((1, 1))
    alerts = np.ones((1, 2))
    game_loop = np.ones((1, 1))
    score_cumulative = np.ones((1, 13))
    score_by_category = np.ones((1, 55))
    score_by_vital = np.ones((1, 9))
    player = np.ones((1, 11))
    control_groups = np.ones((1, 20))
    upgrades = np.ones((1, 1))
    available_actions_one_hot = np.ones((1, 573))

    starcraft_model = total_model()
    # starcraft_model.summary()

    action_type_logits, queued_logits, selected_unit_logits, shifted_logits4, shifted_logits5, control_group_id, minimap_logits, screen1_logits = starcraft_model.predict(x=[a, b, single_select, multi_select, build_queue, cargo,
                                     production_queue, last_actions, cargo_slots_available, home_race_requested,
                                     away_race_requested, action_result, alerts, game_loop,
                                     score_cumulative, score_by_category, score_by_vital,
                                     player, control_groups, upgrades, available_actions_one_hot])

    print(action_type_logits, queued_logits,
                         selected_unit_logits, shifted_logits4, shifted_logits5, control_group_id, minimap_logits, screen1_logits
                         # minimap_logits, screen1_logits, screen2_logits
          )
