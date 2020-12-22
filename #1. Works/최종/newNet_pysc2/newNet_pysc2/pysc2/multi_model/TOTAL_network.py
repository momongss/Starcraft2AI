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

    entity_encoder_model = entity_encoder(entity_list)
    spatial_encoder_model = spatial_encoder(minimap)
    scalar_encoder_model = scalar_encoder(single_select, multi_select, build_queue, cargo, production_queue, last_actions,
                                          cargo_slots_available, home_race_requested, away_race_requested, action_result,
                                          alerts, game_loop, player, control_groups, upgrades, available_actions_one_hot)

    embedded_entity = entity_encoder_model.output['embedded_entity']
    entity_embeddings = entity_encoder_model.output['entity_embeddings']

    map_skip = spatial_encoder_model.output['map_skip']
    embedded_spatial = spatial_encoder_model.output['embedded_spatial']

    embedded_scalar = scalar_encoder_model.output['embedded_scalar']
    scalar_context = scalar_encoder_model.output['scalar_context']

    core_model = core(embedded_scalar, embedded_entity, embedded_spatial)

    lstm_output = core_model.output['lstm_output']

    action_type_encoder_model = action_type_encoder(lstm_output, scalar_context)

    action_type_logits = action_type_encoder_model.output['action_type_logits']
    action_type = action_type_encoder_model.output['action_type']
    autoregressive_embedding = action_type_encoder_model.output['autoregressive_embedding']

    queued_encoder_model = queued_encoder(autoregressive_embedding)
    queued_logits = queued_encoder_model.output['queued_logits']
    queued = queued_encoder_model.output['queued']

    selected_units_encoder_model = selected_units_encoder(autoregressive_embedding, action_type, entity_list)
    selected_unit_logits = selected_units_encoder_model.output['selected_unit_logits']

    target_unit_encoder_model = target_unit_encoder(entity_embeddings, lstm_output)
    shifted_logits4 = target_unit_encoder_model.output['shifted_logits4']
    shifted_logits5 = target_unit_encoder_model.output['shifted_logits5']
    control_group_id = target_unit_encoder_model.output['control_group_id']

    target_point_encoder_model = target_point_encoder(autoregressive_embedding, action_type, map_skip)
    minimap_logits = target_point_encoder_model.output['minimap_logits']
    screen1_logits = target_point_encoder_model.output['screen1_logits']
    screen2_logits = target_point_encoder_model.output['screen2_logits']

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