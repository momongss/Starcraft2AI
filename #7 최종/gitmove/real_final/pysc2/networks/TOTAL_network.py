from pysc2.networks.Action_type import action_type_encoder
from pysc2.networks.Core import core, core_lstm_cell
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
    entity_list = layers.Input(shape=(500, 46), name='entity_list', batch_size=1)
    minimap = layers.Input(shape=(128, 128, 11), name='minimap', batch_size=1)

    # scalar features
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

    embedded_entity, entity_embeddings = entity_encoder(entity_list)
    map_skip, embedded_spatial = spatial_encoder(minimap)
    embedded_scalar, scalar_context = scalar_encoder(single_select, multi_select, build_queue, cargo, production_queue,
                                                     last_actions,
                                                     cargo_slots_available, home_race_requested, away_race_requested,
                                                     action_result,
                                                     alerts, game_loop, player,
                                                     control_groups, upgrades, available_actions_one_hot)

    lstm_output = core(embedded_scalar, embedded_entity, embedded_spatial)
    # lstm_output, next_state_h, next_state_c = core_lstm_cell(embedded_scalar, embedded_entity, embedded_spatial, prev_state_h, prev_state_c)

    action_type_logits, action_type, autoregressive_embedding = action_type_encoder(lstm_output, scalar_context)

    queued_logits, queued = queued_encoder(autoregressive_embedding)

    selected_unit_logits = selected_units_encoder(autoregressive_embedding, action_type, entity_list)

    shifted_logits4, shifted_logits5, control_group_id = target_unit_encoder(entity_embeddings, lstm_output)

    minimap_logits, screen1_logits, screen2_logits = target_point_encoder(autoregressive_embedding, action_type, map_skip)

    starcraft_model = models.Model(inputs=[entity_list, minimap,
                                           single_select, multi_select, build_queue, cargo, production_queue,
                                           last_actions,
                                           cargo_slots_available, home_race_requested, away_race_requested,
                                           action_result,
                                           alerts, game_loop, player,
                                           control_groups, upgrades, available_actions_one_hot],
                                   outputs=[action_type_logits, queued_logits,
                                            selected_unit_logits, shifted_logits4, shifted_logits5, control_group_id,
                                            minimap_logits, screen1_logits, screen2_logits]
                                   )

    starcraft_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.KLDivergence())
    return starcraft_model


if __name__ == '__main__':
    starcraft_model = total_model()
    starcraft_model.summary()
