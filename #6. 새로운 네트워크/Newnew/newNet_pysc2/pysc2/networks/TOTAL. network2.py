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
    lstm_out = layers.LSTM(384, return_state=True)(x)

    # LSTM_model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial], outputs=lstm_out)
    # lstm_out

    starcraft_model = models.Model(inputs=[entity_list, minimap,
                                           single_select, multi_select, build_queue, cargo, production_queue,
                                           last_actions,
                                           cargo_slots_available, home_race_requested, away_race_requested,
                                           action_result,
                                           alerts, game_loop, score_cumulative, score_by_category, score_by_vital,
                                           player,
                                           control_groups, upgrades, available_actions_one_hot, lstm_input],
                                   outputs=[lstm_out])

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

    lstm_input = np.array([0.7608305, 0.0, 0.47068366, 0.9984498, 0.0, 0.4516444, 0.2132346, 0.0, 0.0, 0.5628444, 0.0, 0.0811438, 0.003565073, 0.037921593, 0.19088358, 0.0, 0.34296882, 0.0, 0.0, 0.0, 0.05908689, 0.90554106, 0.0, 0.024107188, 0.0, 0.0, 0.0, 0.34987155, 0.0, 0.32175532, 0.0, 0.36380446, 1.3214751, 0.0, 3.827723, 0.0, 0.0, 0.20181419, 0.03385894, 2.5278873, 0.0, 0.0, 0.0, 0.7662897, 0.0, 0.089543335, 1.5892205, 0.0, 0.0, 0.0, 0.9779957, 1.2707428, 0.0, 0.0, 0.7856329, 0.0, 0.10416472, 0.0, 0.0, 0.0, 0.92435205, 0.0, 0.13626806, 0.8826522, 0.0, 0.0, 1.6608025, 0.7709158, 1.672528, 1.6976124, 0.9710002, 0.0, 0.0, 0.31623977, 0.0, 0.0, 1.3966489, 0.8313209, 2.5290537, 1.0017295, 1.1000547, 0.0, 0.0, 0.7928574, 0.0, 0.0, 0.5798565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7763978, 0.22055826, 0.0, 1.0601711, 1.5445213, 0.0, 0.9426388, 1.3523989, 0.88432914, 0.5084635, 0.0, 0.0, 0.0, 1.2621, 0.2970385, 0.0, 0.0, 0.11092213, 0.0, 0.671426, 0.0, 1.1497207, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.28513956, 0.0, 0.36196932, 0.0, 0.0, 0.0, 0.0, 0.09324322, 0.21224512, 0.0, 1.071265, 0.0, 0.19826484, 0.0, 0.9823066, 1.0852706, 0.0, 0.8343642, 0.0, 0.38066784, 0.0, 0.0, 0.0, 0.14168666, 0.0, 0.28271246, 0.0, 0.43226695, 0.0, 0.32209072, 0.55440325, 0.08345428, 0.0, 0.0, 0.19421461, 0.51302874, 0.35444483, 0.0, 0.025759727, 0.0, 0.0, 0.0, 0.3647415, 0.0, 0.38276005, 0.0, 0.0, 0.3705603, 0.09630746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.054656565, 0.0, 0.26370966, 0.0, 0.40886873, 0.0, 0.18673211, 0.0, 0.0, 0.29685187, 0.0, 0.0, 0.27202225, 0.26571506, 0.109148204, 0.35699898, 0.13462418, 0.0, 0.40990585, 0.18341708, 0.21628278, 0.0, 0.0, 0.02724564, 0.0, 0.37440842, 0.35660434, 0.18824434, 0.11433166, 0.0, 0.34444517, 0.10224259, 0.40963095, 0.0, 0.41361177, 0.31930304, 0.31051028, 0.13820899, 0.0, 0.06737, 0.0, 0.0, 0.11595762, 0.1965788, 0.0, 0.0, 0.0, 0.0, 0.21431506, 0.0, 0.061124712, 0.0, 0.10752934, 0.21787667, 0.0, 0.041609973, 0.0, 0.23677474, 0.0, 0.29202348, 0.3666439, 0.0, 0.08685058, 0.2518263, 0.0, 0.0, 0.10275328, 0.044897318, 0.0, 0.3587656, 0.19918793, 0.3220238, 0.10039461, 0.0, 0.3608641, 0.21651989, 0.10161239, 0.30867958, 0.035256594, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25829566, 0.34678286, 0.3647738, 0.36217403, 0.12428415, 0.4198404, 0.0, 0.0, 0.0, 0.40168834, 0.004580885, 0.0, 0.0, 0.21577764, 0.0, 0.1933443, 0.040726423, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3214919, 0.0, 0.0, 0.13835955, 0.18460107, 0.0, 0.15815365, 0.0, 0.34110916, 0.0, 0.13231617, 0.16136456, 0.0, 0.0, 0.046533853, 0.0, 0.0, 0.0, 0.0, 0.3009379, 0.21219903, 0.27140576, 0.35710472, 0.34439355, 0.19074613, 0.09540677, 0.0, 0.0, 0.0, 0.3308748, 0.0, 0.3266952, 0.0, 0.40736258, 0.0, 0.0, 0.049021333, 0.009147823, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06141016, 0.01656197, 0.37916106, 0.027638406, 0.039013326, 0.0, 0.0, 0.03227049, 0.0, 0.100503385, 0.18296146, 0.27002084, 0.0, 0.0, 0.1867803, 0.0, 0.0, 0.030168861, 0.0, 0.0, 0.0, 0.0034675002, 0.0, 0.0, 0.0, 0.0, 0.2117666, 0.01225242, 0.3303169, 0.0, 0.0, 0.047620416, 0.3526652, 0.0, 0.0, 0.0, 0.0, 0.33549207, 0.04702276, 0.0106986165, 0.27718985, 0.30072093, 0.1607356, 0.0, 0.0, 0.003041029, 0.056564152, 0.10204995, 0.13029069, 0.119485736, 0.0, 0.17669213, 0.01833871, 0.3904574, 0.25162143, 0.0, 0.25311333, 0.41649395, 0.0, 0.0, 1.0801617, 0.0, 0.0, 0.0, 0.272533, 0.02567327, 0.27942875, 0.0, 0.1755192, 1.48791, 1.113584, 1.2883027, 0.0, 0.6361956, 0.21017006, 0.14415112, 0.54849887, 1.22487545e-05, 0.37620175, 0.045990914, 0.6806189, 0.5126596, 0.0, 0.0, 1.4407251, 0.0, 0.77517843, 0.1311651, 0.0, 0.40402728, 0.32986432, 0.0, 1.5013235, 0.0, 0.3567965, 0.61599904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.83778054, 3.6484294, 0.0, 0.5658606, 0.0, 1.0463427, 0.31347123, 0.9590899, 1.5028179, 0.0, 0.0, 0.0, 0.5329938, 0.2550811, 0.0, 1.2382398, 0.518453, 0.010017082, 0.0, 0.18272392, 0.0, 0.0, 0.27213058, 0.0, 0.0, 0.53756416, 0.17915057, 0.09269634, 0.83788335, 0.0, 0.108109415, 0.0, 0.10063654, 1.1517146, 0.47540534, 0.0, 0.0, 0.06474258, 0.0, 0.5271735, 0.57050955, 0.0, 1.1908218, 0.034950078, 0.1171214, 0.3319062, 0.39943817, 0.3339924, 0.0, 0.5126109, 0.0, 0.9283407, 0.0, 0.0, 0.0, 0.09679812, 0.0, 0.2593897, 0.53152937, 0.22944978, 0.0, 0.7040787, 0.6267669, 0.6228124, 0.4678986, 1.1853566, 0.20368883, 0.4449311, 0.41751748, 1.279594, 0.066955894, 0.2718313, 1.3177958, 0.0, 0.7619751, 0.2029241, 0.7577532, 0.0, 0.07373375, 0.0, 0.0, 1.0374243, 0.0, 0.9608772, 0.0, 0.0, 0.0, 0.0, 1.1148545, 0.0, 0.0, 0.0, 0.33165976, 0.0, 1.3167744, 0.28159675, 0.41009068, 0.0, 0.50256, 0.8743589, 1.0002061, 2.2003999, 0.0, 1.0198237, 1.909379, 0.0, 1.220948, 0.0, 0.59945345, 1.6284503, 0.0, 0.4984567, 0.0, 0.0, 0.1415664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03901586, 0.04805717, 0.0, 0.0, 0.35919827, 0.25999343, 0.0, 0.0, 0.0, 0.40592444, 0.0, 0.067675084, 0.0, 0.0, 0.0, 0.28784502, 0.0, 0.25150502, 0.0, 0.0, 0.12724662, 0.099758804, 0.0, 0.0, 0.36292428, 0.1529789, 0.26036394, 0.0, 0.0, 2.0062988, 0.6213415, 0.9418675, 0.21378984, 0.0029010363, 0.79189956, 0.0, 0.0, 2.077618, 0.0, 0.5015221, 0.0, 0.6406153, 0.0, 0.0, 0.0, 0.36642247, 0.0, 2.0362341, 0.0, 0.0, 0.78635514, 2.4970245, 0.0, 0.0, 0.0, 0.70858717, 1.19418, 0.0, 0.7453633, -0.64675236, 1.2931067, -0.49138162, -0.27334893, -0.78421944, 0.29936838, -0.5971621, -0.40261608, -0.89691925, -0.47007105, 0.9608555, 0.3448992, 1.0072019, 0.15648617, 0.87802374, -0.10409944, -0.13666244, -1.1257012, 0.38240746, 0.38357118, 0.33935866, 0.59989834, 0.9213035, 0.27385634, -0.096200064, 0.05750058, -0.52103424, -0.18818885, -0.09928207, -0.5114564, 1.8398888, -0.19380103, 0.059638705, 1.2450242, -1.417711, -0.85644954, -0.1303139, -0.75719553, -0.8037916, -0.8129464, -0.4859262, -0.8748285, -0.6907748, -0.59419394, -0.41704094, -0.082627855, 0.14235228, -0.039223522, 1.6525083, 0.599775, 0.37676972, -0.6970748, -1.1855813, -1.2111276, 0.10396448, 0.43411326, -0.2259881, -0.9516868, -2.0498059, -0.34684035, 0.5367356, -0.5494479, -0.6819617, 0.4060075, 0.40847653, -0.079440564, 0.12144984, 0.2429304, 0.68555075, 0.5532392, -0.64073473, 0.117886424, -0.97335863, 0.63717127, -0.39453736, -0.90633357, 0.5228326, 0.26598248, -0.7272532, 1.7182552, -0.98112154, 0.1631116, -0.20591415, 0.5198444, -0.2225677, -1.3542478, 0.29068702, 1.6719674, 0.35644144, 0.8350136, -1.3820103, 0.8448973, -1.8197109, -0.60399747, -0.5891586, 0.046342686, -0.0131307095, 0.3039086, -0.3453626, -0.5614367, 0.39921442, 0.18482096, 0.09901116, 0.64191717, -0.7663254, 0.845649, -0.15617678, 0.83124965, 0.38201442, 0.38151968, 0.44258004, 1.9011959, 1.1070731, -0.4210414, 0.6551132, 0.8398345, -0.56690425, 0.3711938, 0.32246262, -0.60919505, -0.614442, 0.66920674, -0.07697503, 1.1879603, -0.09667202, -1.0128454, -0.852992, -0.7065351, -0.5261312, -0.5856383, 0.36459196, -0.2771843, 0.6335338, 1.435043, 0.104190946, -0.8828432, -0.4322658, 0.5822359, -0.3152755, 0.2417996, 1.0050241, -0.6629964, 1.7432133, -0.6249705, 0.15591606, 0.86658794, -0.638641, -0.2499826, -0.8005645, -0.8180169, -0.749239, -0.18683359, 1.3034694, 0.25055477, -0.3406219, 1.5250795, -1.2853619, -0.18424404, 0.24328311, -1.0319276, -0.31929126, -0.21193099, 0.8331923, -0.06361447, 0.25658405, -0.39235854, -0.70177585, -1.1766981, -0.8710575, 1.9351304, 1.8042487, -0.4647341, 0.9653151, 0.9866788, 0.27956954, 0.13951316, -2.4777856, -0.14493845, 1.2174962, 1.3836915, 0.1585545, -0.61846733, -0.040082604, -0.007711597, 0.15186217, -0.17779285, 0.5092276, -0.14638375, -0.7337706, 0.2679727, 0.68174106, -0.21115752, 0.26072, -1.4188497, -0.9207305, -0.303776, 1.135449, 0.9478971, 0.27039972, 0.46505868, -0.59227514, 0.23631456, 0.6468738, 1.0896316, -1.5851865, 0.17772755, 0.72526634, -1.0347873, 1.6838098, -0.34892154, 1.100978, 0.53967106, -0.8794065, -0.69566554, -0.17261362, -0.57193196, 0.8148756, 0.13593782, -0.29070467, -0.33103198, 1.1967382, 0.37486, -0.23165716, 0.45205244, 1.365395, -1.165448, -0.28904608, 0.48287678, 0.27934968, -0.72012496, -0.9254907, -0.7149423, -0.28797543, 0.84176236, -0.082627594, 0.4474954, 0.31713706, 0.5620291, 0.75767833, -1.0320209, -0.3014316, 0.7607186, 0.35342407, 1.1037279, 0.91802394, -0.36657208, 1.0875882, 0.10361484, 0.030043323, -0.79844, -0.72436404, 0.94233567, 0.25848818, 0.55676633, -0.3475282, 0.7198541, 0.0, 0.5869567, 1.091787, 1.7434464, 0.0, 1.0776248, 0.0, 1.0102533, 0.13639006, 0.0, 0.16564013, 0.78550225, 0.6701685, 0.6139077, 0.0, 0.0, 0.50745994, 0.36888003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.074868776, 0.0, 0.0, 0.0, 0.0, 2.7205477, 0.2734733, 1.4592838, 0.0, 0.0, 0.65195733, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52675617, 0.9993315, 0.90293235, 0.7184663, 1.1960233, 1.1090053, 0.0, 0.7962969, 0.0, 0.8218665, 0.0, 0.0, 0.0, 0.0, 0.33769113, 0.0, 0.7798124, 0.8244232, 0.0, 0.0, 0.0, 0.0, 0.62634236, 0.0, 0.37468943, 0.5677697, 0.5406906, 0.9491582, 0.17929941, 0.0, 1.7405305, 0.0, 0.0, 0.3615444, 0.0, 1.2146385, 0.0, 0.0, 0.0, 1.0528295, 0.904494, 0.172399, 0.4442695, 1.995485, 0.0, 0.0, 0.0, 1.6456718, 0.46452144, 0.0, 0.102437824, 0.19587845, 0.32746372, 0.0, 0.84106463, 0.0, 1.1119252, 0.9404204, 0.888969, 0.77385944, 0.0, 0.56695527, 1.4668862, 0.0, 0.62976426, 0.98591524, 0.0, 0.67874444, 0.19305697, 0.2489702, 0.0, 0.0, 0.0, 1.0776742, 0.0, 1.9976239, 0.0, 0.0, 0.0, 0.7441056, 0.0, 0.0, 0.0, 0.6865031, 0.0, 1.2494447, 0.72730464, 0.15152782, 0.0, 0.0, 1.2881123, 0.8649372, 0.31247732, 0.0, 0.0, 0.8775394, 1.1259401, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0573523, 0.34104434, 0.0, 0.15044525, 0.0, 0.6941989, 0.49685726, 1.2826319, 2.0238547, 0.0, 0.0, 1.2864242, 0.0, 0.7809141, 0.0, 1.1985584, 0.22247142, 0.2678286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57258976, 0.0, 0.0, 0.65942425, 0.0, 0.0, 0.0, 0.5157379, 0.0, 0.0, 1.6837095, 0.0, 1.2925534, 0.0, 0.3045206, 0.33921057, 0.0, 0.04687529, 0.9199721, 2.5468893, 0.0, 2.6690283, 0.0, 0.0, 1.1491294, 2.009449, 0.0, 1.1810656, 0.0, 0.0, 0.0, 0.0, 1.0795408, 0.0, 0.4767682, 0.0, 0.0, 0.0, 1.5437323, 1.2681729, 0.4322838, 0.21094187, 0.14243346, 0.0, 0.0, 0.9054004, 0.9609903, 0.0, 0.0, 0.0, 0.20659967, 1.5581335, 0.26227927, 1.5792596, 0.0, 1.5326637, 0.0, 0.0, 0.0, 0.6614867, 1.2588065, 0.0, 0.011782748, 0.3256714, 0.0, 0.0, 0.9896995, 0.0, 0.0, 0.0, 1.9464437, 1.7017078, 0.0, 0.29759955, 0.3284374, 1.1127996, 0.93230504, 0.0, 0.19486834, 0.0, 0.24350554, 0.0, 0.5019626, 0.0, 0.08351589, 0.0, 0.0, 0.0614143, 0.0])
    lstm_input = lstm_input.reshape(1, 1120)
    print(lstm_input.shape)
    lstm_out = starcraft_model.predict(x=[a, b, single_select, multi_select, build_queue, cargo,
                                     production_queue, last_actions, cargo_slots_available, home_race_requested,
                                     away_race_requested, action_result, alerts, game_loop,
                                     score_cumulative, score_by_category, score_by_vital,
                                     player, control_groups, upgrades, available_actions_one_hot, lstm_input])

    a = list(lstm_out[0])
    print(a)
    # print(embedded_scalar.shape)
    # print(embedded_entity.shape)
    # print(embedded_spatial.shape)

    # print(embedded_scalar.shape, embedded_entity.shape, embedded_spatial.shape)