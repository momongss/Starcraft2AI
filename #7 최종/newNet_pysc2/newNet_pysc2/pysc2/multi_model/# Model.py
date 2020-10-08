from pysc2.multi_model.Action_type import action_type_encoder
from pysc2.multi_model.Core import core, core_lstm_cell
from pysc2.multi_model.Entitiy_encoder import entity_encoder
from pysc2.multi_model.Queued import queued_encoder
from pysc2.multi_model.Scalar_encoder import scalar_encoder
from pysc2.multi_model.Selected_units import selected_units_encoder
from pysc2.multi_model.Spatial_encoder import spatial_encoder
from pysc2.multi_model.Target_unit import target_unit_encoder
from pysc2.multi_model.Target_point import target_point_encoder

from tensorflow import keras
from tensorflow.keras import layers, models


def total_model():
    entity_list = layers.Input(shape=(500, 46), name='entity_list', batch_size=1)
    minimap = layers.Input(shape=(128, 128, 11), name='minimap', batch_size=1)

    single_select = layers.Input(shape=(7,), name='single_select', batch_size=1)
    multi_select = layers.Input(shape=(140,), name='multi_select', batch_size=1)
    build_queue = layers.Input(shape=(112,), name='build_queue', batch_size=1)
    cargo = layers.Input(shape=(112,), name='cargo', batch_size=1)
    production_queue = layers.Input(shape=(16,), name='production_queue', batch_size=1)
    last_actions = layers.Input(shape=(1,), name='last_actions', batch_size=1)
    cargo_slots_available = layers.Input(shape=(1,), name='cargo_slots_available', batch_size=1)
    home_race_requested = layers.Input(shape=(1,), name='home_race_requested', batch_size=1)
    away_race_requested = layers.Input(shape=(1,), name='away_race_requested', batch_size=1)
    action_result = layers.Input(shape=(1,), name='action_result', batch_size=1)
    alerts = layers.Input(shape=(2,), name='alerts', batch_size=1)
    game_loop = layers.Input(shape=(1,), name='game_loop', batch_size=1)
    player = layers.Input(shape=(11,), name='player', batch_size=1)
    control_groups = layers.Input(shape=(20,), name='control_groups', batch_size=1)
    upgrades = layers.Input(shape=(1,), name='upgrades', batch_size=1)
    available_actions_one_hot = layers.Input(shape=(573,), name='available_actions_one_hot', batch_size=1)

    # Entity_encoder
    entity_flatten = layers.Flatten()(entity_list)

    embedded_entity = layers.ReLU()(entity_list)
    embedded_entity = layers.Conv1D(filters=256, kernel_size=1)(embedded_entity)
    embedded_entity = layers.Flatten()(embedded_entity)

    embedded_entity = layers.Dense(256)(embedded_entity)
    embedded_entity = layers.Dense(256)(embedded_entity)

    entity_embeddings = layers.Dense(256, activation='relu')(entity_flatten)
    entity_embeddings = layers.Flatten()(entity_embeddings)

    entity_encoder_model = models.Model(inputs=entity_list,
                                        outputs={'embedded_entity': embedded_entity,
                                                 'entity_embeddings': entity_embeddings},
                                        name='Entity_encoder')

    # Spatial_encoder
    resblock_channels = 128
    resblock_kernel = 3
    resblock_layer = 4

    # minimap = layers.Input(shape=(128, 128, 11))
    c = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(minimap)
    c = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(c)
    c = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(c)
    map_skip = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(c)

    for _ in range(resblock_layer):
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(map_skip)
        r = layers.BatchNormalization()(r)
        r = layers.Activation('relu')(r)
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(r)
        r = layers.BatchNormalization()(r)
        map_skip = layers.add([r, map_skip])

    map_flatten = layers.Flatten()(map_skip)
    embedded_spatial = layers.Dense(256, activation='relu', name='embedded_spatial')(map_flatten)
    embedded_spatial = layers.Flatten()(embedded_spatial)

    spatial_encoder_model = models.Model(inputs=minimap,
                                         outputs={'map_skip': map_skip,
                                                  'embedded_spatial': embedded_spatial},
                                         name='Spatial_encoder')

    # Scalar_encoder
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
    ])

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
                                          player_out,
                                          control_groups_out,
                                          upgrades_out,
                                          available_actions_one_hot_out])

    embedded_scalar = layers.Flatten()(embedded_scalar)

    scalar_encoder_model = models.Model(inputs=[single_select, multi_select, build_queue, cargo, production_queue,
                                                last_actions, cargo_slots_available, home_race_requested,
                                                away_race_requested, action_result, alerts, game_loop, player,
                                                control_groups, upgrades, available_actions_one_hot],
                                        outputs={'embedded_scalar': embedded_scalar,
                                                 'scalar_context': scalar_context})



    starcraft_model = models.Model(inputs=[entity_list, minimap],
                                   outputs=[action_type_logits, queued_logits,
                                            selected_unit_logits, shifted_logits4, shifted_logits5, control_group_id,
                                            minimap_logits, screen1_logits, screen2_logits])

    starcraft_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-7), loss=keras.losses.KLDivergence(),
                            loss_weights={'action_type_logits': 2.0,
                                          'queued_logits': 0.5,
                                          'selected_unit_logits': 1.0,
                                          'shifted_logits4': 0.1,
                                          'shifted_logits5': 0.1,
                                          'control_group_id': 0.1,
                                          'minimap_logits': 1.0,
                                          'screen1_logits': 1.0,
                                          'screen2_logits': 1.0})
    return starcraft_model


if __name__ == '__main__':
    starcraft_model = total_model()
    starcraft_model.summary()