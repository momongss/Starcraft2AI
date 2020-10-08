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
import tensorflow as tf


tf.debugging.set_log_device_placement(True)

def total_model():
    entity_list = layers.Input(shape=(500, 46), name='entity_list')
    minimap = layers.Input(shape=(128, 128, 11), name='minimap')
    # embedded_scalar = layers.Input(shape=(256,), name='embedded_scalar')


    # scalar features

    embedded_entity, entity_embeddings = entity_encoder(entity_list)
    map_skip, embedded_spatial = spatial_encoder(minimap)

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

    embedded_scalar, scalar_context = scalar_encoder(single_select, multi_select, build_queue, cargo, production_queue,
                                                     last_actions,
                                                     cargo_slots_available, home_race_requested, away_race_requested,
                                                     action_result,
                                                     alerts, game_loop, score_cumulative, score_by_category,
                                                     score_by_vital, player,
                                                     control_groups, upgrades, available_actions_one_hot)

    lstm_output = core(embedded_scalar, embedded_entity, embedded_spatial)
    print(lstm_output)

    starcraft_model = models.Model(inputs=[entity_list, minimap, single_select, multi_select, build_queue, cargo, production_queue,
                                             last_actions,
                                             cargo_slots_available, home_race_requested, away_race_requested,
                                             action_result,
                                             alerts, game_loop, score_cumulative, score_by_category,
                                             score_by_vital, player,
                                             control_groups, upgrades, available_actions_one_hot],
                                   outputs=lstm_output)

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

    lstm_output = starcraft_model.predict(x=[a, b, single_select, multi_select, build_queue, cargo,
                                     production_queue, last_actions, cargo_slots_available, home_race_requested,
                                     away_race_requested, action_result, alerts, game_loop,
                                     score_cumulative, score_by_category, score_by_vital,
                                     player, control_groups, upgrades, available_actions_one_hot])

    # print(embedded_scalar.shape, embedded_entity.shape, embedded_spatial.shape)