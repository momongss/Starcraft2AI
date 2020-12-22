from tensorflow import keras
from tensorflow.keras import layers, models


def total_model():
    entity_list = layers.Input(shape=(500, 46), name='entity_list')
    minimap = layers.Input(shape=(128, 128, 11), name='minimap')

    # scalar features
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

    # scalar encoder
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
    # embedded_scalar, scalar_context

    # spatial encoder
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

    # map_skip, embedded_spatial

    # entity encoder
    entity_flatten = layers.Flatten()(entity_list)

    embedded_entity = layers.ReLU()(entity_list)
    embedded_entity = layers.Conv1D(filters=256, kernel_size=1)(embedded_entity)
    embedded_entity = layers.Flatten()(embedded_entity)

    embedded_entity = layers.Dense(256)(embedded_entity)
    embedded_entity = layers.Dense(256)(embedded_entity)

    entity_embeddings = layers.Dense(256, activation='relu')(entity_flatten)
    # model = models.Model(inputs=entity_list, outputs=[embedded_entity, entity_embeddings], name='entity_model')
    # embedded_entity, entity_embeddings

    # core encoder
    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(x)
    x = layers.LSTM(384, return_sequences=True)(x)
    lstm_out = layers.LSTM(384)(x)

    # LSTM_model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial], outputs=lstm_out)
    # lstm_out

    # action_type encoder
    conv_width = 384
    action_type_len = 572

    # lstm_output = keras.layers.Input(shape=(384, ), name='lstm_output')
    # scalar_context = keras.Input(shape=(160,), name='scalar_context')

    Resblock_output = layers.Reshape((1, 384), input_shape=(384,))(lstm_out)

    Resblock_layer = 16

    for _ in range(Resblock_layer):
        x = layers.Conv1D(conv_width, 1, padding='valid', strides=1)(Resblock_output)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(conv_width, 1, padding='valid', strides=1)(x)
        x = layers.BatchNormalization()(x)
        Resblock_output = layers.add([x, Resblock_output])

    Resblock_flatten = layers.Flatten()(Resblock_output)
    gate = layers.Dense(conv_width, activation='sigmoid')(scalar_context)
    gate_input = layers.multiply([Resblock_flatten, gate])
    action_type_logits = layers.Dense(action_type_len, activation='sigmoid', name='action_type_logits')(gate_input)

    action_type = keras.backend.argmax(action_type_logits)
    action_type_one_hot = keras.backend.one_hot(action_type, action_type_len)

    gate_input = layers.Dense(256, activation='relu')(action_type_one_hot)
    gate = layers.Dense(256, activation='sigmoid')(scalar_context)
    gate_input = layers.multiply([gate_input, gate])
    autoregressive_embedding = layers.Dense(1024)(gate_input)

    gate = layers.Dense(384, activation='sigmoid')(scalar_context)
    gate_input = layers.multiply([lstm_out, gate])
    lstm_projection = layers.Dense(1024)(gate_input)

    autoregressive_embedding = layers.add([autoregressive_embedding, lstm_projection], name='autoregressive_embedding')

    # model = models.Model(inputs=[lstm_output, scalar_context], outputs=[action_type_logits, action_type, autoregressive_embedding], name='action_type(resnet)')
    # action_type_logits, action_type, autoregressive_embedding

    # queued encoder
    layer_width = 256

    # autoregressive_embedding = keras.Input(shape=(1024,), name='Queued_input')

    x = layers.Dense(layer_width, activation='relu')(autoregressive_embedding)
    x = layers.Dense(layer_width, activation='relu')(x)
    queued_logits = layers.Dense(2, activation='softmax', name='queued_logits')(x)
    queued = keras.backend.argmax(queued_logits)

    # model = models.Model(inputs=autoregressive_embedding, outputs=[queued_logits, queued], name='Queued_output')
    # queued_logits, queued

    # selected_units encoder
    enc = layers.Dense(1)(entity_list)
    enc = layers.Flatten()(enc)

    dec = layers.Dense(500)(autoregressive_embedding)

    add = layers.add([enc, dec])
    tanh = keras.activations.tanh(add)
    selected_unit_logits = layers.Dense(500)(tanh)

    # model = models.Model(inputs=[autoregressive_embedding, action_type, entity_list], outputs=unit_logits,
    #                      name='action_type_output')
    # unit_logits

    # target_unit encoder
    layer_width = 256

    # entity_embeddings = keras.Input(shape=(256,), name='entity_embeddings')
    # lstm_out = keras.Input(shape=(384,), name='Queued_input')

    shifted_input = layers.concatenate([entity_embeddings, lstm_out])
    x = layers.Dense(layer_width, activation='relu')(shifted_input)
    x = layers.Dense(layer_width, activation='relu')(x)
    shifted_logits = layers.Dense(5, activation='softmax', name='shifted_logits')(x)

    x = layers.Dense(layer_width, activation='relu')(shifted_input)
    x = layers.Dense(layer_width, activation='relu')(x)
    control_group_id = layers.Dense(10, activation='softmax', name='control_group_id')(x)

    # model = models.Model(inputs=[entity_embeddings, lstm_out], outputs=[shifted_logits, control_group_id], name='target_unit')
    # shifted_logits, control_group_id

    # target_point encoder
    resblock_channels = 128
    resblock_kernel = 3
    resblock_layer = 4

    autoregressive_embedding = layers.Input(shape=(1024,))
    action_type = layers.Input(shape=(1,))
    map_skip = layers.Input(shape=(16, 16, 128))

    embedding_skip = layers.Reshape(target_shape=(16, 16, 4))(autoregressive_embedding)
    map_skip_input = layers.concatenate([map_skip, embedding_skip])

    map_relu = keras.activations.relu(map_skip_input)
    map_conv = layers.Conv2D(128, 1, activation='relu')(map_relu)

    for _ in range(resblock_layer):
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(map_conv)
        r = layers.BatchNormalization()(r)
        r = layers.Activation('relu')(r)
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(r)
        r = layers.BatchNormalization()(r)
        map_conv = layers.add([r, map_skip])

    map_conv = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(16, 4, strides=2, padding='same')(map_conv)
    minimap_logits = layers.Conv2DTranspose(1, 4, strides=1, padding='same')(map_conv)
    minimap_logits = keras.layers.Reshape((128, 128))(minimap_logits)

    map_conv = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(16, 4, strides=1, padding='same')(map_conv)
    screen1_logits = layers.Conv2DTranspose(1, 4, strides=1, padding='same')(map_conv)
    screen1_logits = keras.layers.Reshape((84, 84))(screen1_logits)

    map_conv = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(16, 4, strides=1, padding='same')(map_conv)
    screen2_logits = layers.Conv2DTranspose(1, 4, strides=1, padding='same')(map_conv)
    screen2_logits = keras.layers.Reshape((84, 84))(screen2_logits)

    # model = models.Model(inputs=[autoregressive_embedding, action_type, map_skip], outputs=[minimap_logits, screen1_logits, screen2_logits], name='spatial_encoder')
    # minimap_logits, screen1_logits, screen2_logits

    starcraft_model = models.Model(inputs=[entity_list, minimap,
                                           single_select, multi_select, build_queue, cargo, production_queue,
                                           last_actions,
                                           cargo_slots_available, home_race_requested, away_race_requested,
                                           action_result,
                                           alerts, game_loop, score_cumulative, score_by_category, score_by_vital,
                                           player,
                                           control_groups, upgrades, available_actions_one_hot],
                                   outputs=[action_type, queued, selected_unit_logits,
                                            shifted_logits, control_group_id,
                                            minimap_logits, screen1_logits, screen2_logits])

    starcraft_model.compile(optimizer=keras.optimizers.Adam, loss=keras.losses.KLDivergence)
    starcraft_model.summary()

total_model()