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

results = []

def train_one_game():
    correct = 0
    incorrect = 0
    for game_loop in sorted(replay_data.keys()):
        feature_screen = normalize(replay_data[game_loop][OBS]['feature_screen'])
        feature_minimap = normalize(replay_data[game_loop][OBS]['feature_minimap'])
        player = normalize(replay_data[game_loop][OBS]['player'])
        normalized_game_loop = np.array([game_loop / (22.4 * 1200)])

        action_id = replay_data[game_loop][ID]

        feature_screen = feature_screen.reshape((1, 128, 128, 27))

        if action_id == 1:
            continue

        action_type_logits = np.zeros(10)
        action_type_logits[action_map[action_id]] = 1
        normalized_game_loop = normalized_game_loop.reshape((1, 1))
        action_type_logits = np.array(action_type_logits).reshape((1, 10))

        print(normalized_game_loop.shape, "s")
        good_model.fit(normalized_game_loop, action_type_logits)

        action_logit = good_model.predict(x={
            'game_loop': normalized_game_loop
        })
        if reverse_action_map[np.argmax(action_logit)] == action_id:
            correct += 1
        else:
            incorrect += 1

    results.append(correct / (correct + incorrect))


for _ in range(10):
    train_one_game()

print(results)
good_model.save_weights('./checkpoints/good_checkpoint')