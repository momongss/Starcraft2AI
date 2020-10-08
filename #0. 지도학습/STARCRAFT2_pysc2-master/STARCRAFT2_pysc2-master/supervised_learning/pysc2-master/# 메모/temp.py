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

single_select <class 'numpy.ndarray'>
multi_select <class 'numpy.ndarray'>
build_queue <class 'numpy.ndarray'>
cargo <class 'numpy.ndarray'>
production_queue <class 'numpy.ndarray'>
last_actions <class 'numpy.ndarray'>
cargo_slots_available <class 'numpy.ndarray'>
home_race_requested <class 'numpy.ndarray'>
away_race_requested <class 'numpy.ndarray'>
map_name <class 'str'>
feature_screen <class 'pysc2.lib.named_array.NamedNumpyArray'>
feature_minimap <class 'pysc2.lib.named_array.NamedNumpyArray'>
action_result <class 'numpy.ndarray'>
alerts <class 'numpy.ndarray'>
game_loop <class 'numpy.ndarray'>
score_cumulative <class 'pysc2.lib.named_array.NamedNumpyArray'>
score_by_category <class 'pysc2.lib.named_array.NamedNumpyArray'>
score_by_vital <class 'pysc2.lib.named_array.NamedNumpyArray'>
player <class 'pysc2.lib.named_array.NamedNumpyArray'>
control_groups <class 'numpy.ndarray'>
feature_units <class 'pysc2.lib.named_array.NamedNumpyArray'>
feature_effects <class 'pysc2.lib.named_array.NamedNumpyArray'>
upgrades <class 'numpy.ndarray'>
available_actions <class 'numpy.ndarray'>
radar <class 'pysc2.lib.named_array.NamedNumpyArray'>
available_actions_one_hot <class 'numpy.ndarray'>