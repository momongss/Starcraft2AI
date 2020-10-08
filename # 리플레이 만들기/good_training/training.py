import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
from architecture import get_compiled_Model

with open('good_replay.txt', 'rb') as fk:
    replay_data = pickle.load(fk)

ID = 0
ARGS = 1
OBS = 2


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


good_model = get_compiled_Model()
# good_model.load_weights('./checkpoints/good_checkpoint')

action_map = {1: 0, 2: 1, 451: 2, 3: 3, 7: 4, 490: 5, 42: 6, 13: 7, 91: 8, 477: 9}
reverse_action_map = [1, 2, 451, 3, 7, 490, 42, 13, 91, 477]

good_logs = keras.callbacks.TensorBoard('.\\tensorboard_logs\\zgood_logs',
                                                   update_freq=1000)

def train_one_game():
    for game_loop in sorted(replay_data.keys()):
        feature_screen = normalize(replay_data[game_loop][OBS]['feature_screen'])
        feature_minimap = normalize(replay_data[game_loop][OBS]['feature_minimap'])
        player = normalize(replay_data[game_loop][OBS]['player'])
        normalized_game_loop = np.array([game_loop / (22.4 * 1200)])

        action_id = replay_data[game_loop][ID]
        args = replay_data[game_loop][ARGS]

        feature_screen = feature_screen.reshape((1, 128, 128, 27))

        if action_id == 1:
            continue

        action_type_logits, screen_x1, screen_y1, screen_x2, screen_y2, minimap_x, minimap_y = good_model.predict(x={
            'feature_screen': feature_screen,
            'feature_minimap': feature_minimap,
            'player': player,
            'game_loop': normalized_game_loop
        })

        action_type_logits = np.zeros(10)
        action_type_logits[action_map[action_id]] = 1
        if action_id == 2:
            screen_x1 = np.zeros(128)
            screen_x1[args[1][0]] = 1
            screen_y1 = np.zeros(128)
            screen_y1[args[1][1]] = 1
        elif action_id == 451:
            screen_x1 = np.zeros(128)
            screen_x1[args[1][0]] = 1
            screen_y1 = np.zeros(128)
            screen_y1[args[1][1]] = 1
        elif action_id == 3:
            screen_x1 = np.zeros(128)
            screen_x1[args[1][0]] = 1
            screen_y1 = np.zeros(128)
            screen_y1[args[1][1]] = 1
            screen_x2 = np.zeros(128)
            screen_x2[args[2][0]] = 1
            screen_y2 = np.zeros(128)
            screen_y2[args[2][1]] = 1
        elif action_id == 7:
            pass
        elif action_id == 490:
            pass
        elif action_id == 42:
            screen_x1 = np.zeros(128)
            screen_x1[args[1][0]] = 1
            screen_y1 = np.zeros(128)
            screen_y1[args[1][1]] = 1
        elif action_id == 13:
            minimap_x = np.zeros(128)
            minimap_x[args[1][0]] = 1
            minimap_y = np.zeros(128)
            minimap_y[args[1][1]] = 1
        elif action_id == 91:
            screen_x1 = np.zeros(128)
            screen_x1[args[1][0]] = 1
            screen_y1 = np.zeros(128)
            screen_y1[args[1][1]] = 1

        action_type_logits = action_type_logits.reshape((1, 10))

        screen_x1 = screen_x1.reshape((1, 128))
        screen_y1 = screen_y1.reshape((1, 128))
        screen_x2 = screen_x2.reshape((1, 128))
        screen_y2 = screen_y2.reshape((1, 128))

        minimap_x = minimap_x.reshape((1, 128))
        minimap_y = minimap_y.reshape((1, 128))

        good_model.fit(x={
            'feature_screen': feature_screen,
            'feature_minimap': feature_minimap,
            'player': player,
            'game_loop': normalized_game_loop
        }, y={
            'action_type_logits': action_type_logits,
            'screen_x1': screen_x1,
            'screen_y1': screen_y1,
            'screen_x2': screen_x2,
            'screen_y2': screen_y2,
            'minimap_x': minimap_x,
            'minimap_y': minimap_y
        },
        callbacks=[good_logs])

for _ in range(10):
    train_one_game()

good_model.save_weights('./checkpoints/good_checkpoint')