# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
from time import time

from tensorflow import keras

from pysc2.agents import base_agent
from pysc2.lib import actions

MAX_UNITS = 50
MAX_POPULATION = 20
MAX_BUILD = 16
MAX_CARGO = 16
MAX_PRODUCTION = 8

max_len_mapping = {'multi_select': MAX_POPULATION,
                   'build_queue': MAX_BUILD,
                   'cargo': MAX_CARGO,
                   'production_queue': MAX_PRODUCTION,
                   'feature_units': MAX_UNITS}

total_act = {'action_type_logits',
             'queued_logits',
             'selected_unit_logits',
             'shifted_logits4',
             'shifted_logits5',
             'control_group_id',
             'minimap_logits',
             'screen1_logits',
             'screen2_logits'}

scalar_obs = [
    # 'feature_units',
    # 'feature_minimap',
    'single_select',
    'multi_select',
    'build_queue',
    'cargo',
    'production_queue',
    'last_actions',
    'cargo_slots_available',
    'home_race_requested',
    'away_race_requested',
    'action_result',
    'alerts',
    'game_loop',
    'score_cumulative',
    'score_by_category',
    'score_by_vital',
    'player',
    'control_groups',
    'upgrades',
    'available_actions_one_hot']

total_obs = [
    'feature_units',
    'feature_minimap',
    'single_select',
    'multi_select',
    'build_queue',
    'cargo',
    'production_queue',
    'last_actions',
    'cargo_slots_available',
    'home_race_requested',
    'away_race_requested',
    'action_result',
    'alerts',
    'game_loop',
    'score_cumulative',
    'score_by_category',
    'score_by_vital',
    'player',
    'control_groups',
    'upgrades',
    'available_actions_one_hot']

# screen_map =

reshape_map = {
    'feature_units': (1, MAX_UNITS, 46),
    'feature_minimap': (1, 128, 128, 11),
    'single_select': (1, 7),
    'multi_select': (1, MAX_POPULATION*7),
    'build_queue': (1, MAX_BUILD*7),
    'cargo': (1, MAX_CARGO*7),
    'production_queue': (1, MAX_PRODUCTION*2),
    'last_actions': (1, 1),
    'cargo_slots_available': (1, 1),
    'home_race_requested': (1, 1),
    'away_race_requested': (1, 1),
    'action_result': (1, 1),
    'alerts': (1, 2),
    'game_loop': (1, 1),
    'score_cumulative': (1, 13),
    'score_by_category': (1, 55),
    'score_by_vital': (1, 9),
    'player': (1, 11),
    'control_groups': (1, 20),
    'upgrades': (1, 1),
    'available_actions_one_hot': (1, 573)
}


def select_function(available_actions):
    return np.random.choice(available_actions)


def select_args(size):
    return np.random.randint(0, size)


def obs_pre_processing(observation):
    def pad_2d(observation, name):
        padding = max_len_mapping[name] - len(observation[name])
        if padding <= 0:
            observation[name] = observation[name][:max_len_mapping[name]]
        # elif padding < max_len_mapping[name]:
        else:
            observation[name] = np.pad(observation[name], ((0, padding), (0, 0)), mode='constant')

    def pad_4_units(observation, name):
        padding = max_len_mapping[name] - len(observation[name])
        if padding <= 0:
            observation[name] = observation[name][:max_len_mapping[name]]
        elif padding < MAX_UNITS:
            observation[name] = np.pad(observation[name], ((0, padding), (0, 0)), mode='constant')
        else:
            observation[name] = np.zeros((MAX_UNITS, 46))

    def pad_1d(observation, name, max_len):
        padding = max_len - observation[name].size
        if padding < 0:
            observation[name] = observation[name][:max_len]
        else:
            observation[name] = np.pad(observation[name], (0, padding), mode='constant')

    if observation['single_select'].shape == (0, 7):  # single_select : (1, 7) 고정
        observation['single_select'] = np.pad(observation['single_select'], ((0, 1), (0, 0)), mode='constant')
    pad_4_units(observation, 'feature_units')
    pad_2d(observation, 'multi_select')         # multi_select : (MAX_POPULATION, 7) 고정
    pad_2d(observation, 'build_queue')          # build_queue : (MAX_BUILD, 7) 고정
    pad_2d(observation, 'cargo')                # cargo (MAX_CARGO, 7) 고정
    pad_2d(observation, 'production_queue')     # production_queue (MAX_PRODUCTION, 2) 고정

    pad_1d(observation, 'last_actions', 1)  # last_actions (1,) 고정
    pad_1d(observation, 'alerts', 2)  # alerts
    pad_1d(observation, 'action_result', 1)  # action_result
    pad_1d(observation, 'upgrades', 1)  # upgrades

    observation['available_actions_one_hot'] = np.array(
        [0 for _ in range(573)], dtype=np.int32)
    for i in observation["available_actions"]:
        observation['available_actions_one_hot'][i] = 1

    return observation


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


# tensorboard_callback = keras.callbacks.TensorBoard('C:\\Users\\rkrp1\\Desktop\\종합설계\\tensorboard_logs\\logs',
#                                                    update_freq=1000)
LSTM1_logs = keras.callbacks.TensorBoard('./tensorboard_logs/LSTM1_logs',
                                                   update_freq=1000)
LSTM2_logs = keras.callbacks.TensorBoard('./tensorboard_logs/LSTM2_logs',
                                                   update_freq=1000)
LSTM3_logs = keras.callbacks.TensorBoard('./tensorboard_logs/LSTM3_logs',
                                                   update_freq=1000)

BI_LSTM1_logs = keras.callbacks.TensorBoard('./tensorboard_logs/BI_LSTM1_logs',
                                                   update_freq=1000)
BI_LSTM2_logs = keras.callbacks.TensorBoard('./tensorboard_logs/BI_LSTM2_logs',
                                                   update_freq=1000)
BI_LSTM3_logs = keras.callbacks.TensorBoard('./tensorboard_logs/BI_LSTM3_logs',
                                                   update_freq=1000)


class ModelAgent(base_agent.BaseAgent):
    def small_update_play(self, action, obs):
        if action == 0:
            return
        LRChangePoint = int(180 * 22.4)
        self.update_count += 1
        if self.update_count == LRChangePoint:
            print('게임시작 3분 이후는 학습하지 않으므로 종료함')
            sys.exit()

        maxValue = 1
        minValue = 0

        obs = obs_pre_processing(obs)

        for key in total_obs:
            obs[key] = np.asarray(obs[key].reshape(reshape_map[key]))

        action_type_logits = np.zeros(573)
        action_type_logits[action] = maxValue

        fs = time()

        obs['feature_units'] = normalize(obs['feature_units'])
        # obs['game_loop'] = normalize(obs['game_loop'])
        obs['game_loop'] = obs['game_loop'] / (22.4*1200)
        obs['player'] = normalize(obs['player'])

        # print("game_loop", obs['game_loop'])

        # w = self.small_model.layers[-1].get_weights()
        # print(w)

        self.LSTM1.fit(
            x={
                'entity_list': obs['feature_units']
                # 'game_loop': obs['game_loop'],
                # 'player': obs['player'],
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0)
            },
            callbacks=[LSTM1_logs]
        )

        self.LSTM2.fit(
            x={
                'entity_list': obs['feature_units']
                # 'game_loop': obs['game_loop'],
                # 'player': obs['player'],
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0)
            },
            callbacks=[LSTM2_logs]
        )

        self.LSTM3.fit(
            x={
                'entity_list': obs['feature_units']
                # 'game_loop': obs['game_loop'],
                # 'player': obs['player'],
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0)
            },
            callbacks=[LSTM3_logs]
        )

        self.BI_LSTM1.fit(
            x={
                'entity_list': obs['feature_units']
                # 'game_loop': obs['game_loop'],
                # 'player': obs['player'],
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0)
            },
            callbacks=[BI_LSTM1_logs]
        )

        self.BI_LSTM2.fit(
            x={
                'entity_list': obs['feature_units']
                # 'game_loop': obs['game_loop'],
                # 'player': obs['player'],
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0)
            },
            callbacks=[BI_LSTM2_logs]
        )

        self.BI_LSTM3.fit(
            x={
                'entity_list': obs['feature_units']
                # 'game_loop': obs['game_loop'],
                # 'player': obs['player'],
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0)
            },
            callbacks=[BI_LSTM3_logs]
        )

        # action_logits = self.small_model.predict(
        #     x={
        #         'entity_list': obs['feature_units'],
        #         'game_loop': obs['game_loop'],
        #         'player': obs['player'],
        #     }
        # )
        # print(action_logits[0][:10])
        # print(np.argmax(action_logits[0]), action)
        fd = time()
        # print('훈련시간', fs - fd)

    def model_update_play(self, action, arg, obs):
        if action == 0:
            return
        LRChangePoint = int(180 * 22.4)
        self.update_count += 1
        if self.update_count == LRChangePoint:
            print('게임시작 3분 이후는 학습하지 않으므로 종료함')
            sys.exit()

        time1 = time()
        maxValue = 1
        minValue = 0

        obs = obs_pre_processing(obs)

        for key in total_obs:
            obs[key] = np.asarray(obs[key].reshape(reshape_map[key])) / 20.0

        # for key in scalar_obs:
        #     obs[key] = obs[key].flatten()

        time2 = time()

        network_output = self.startcraft_model.predict(x={
            'entity_list': obs['feature_units'],
            'minimap': obs['feature_minimap'],
            'single_select': obs['single_select'],
            'multi_select': obs['multi_select'],
            'build_queue': obs['build_queue'],
            'cargo': obs['cargo'],
            'production_queue': obs['production_queue'],
            'last_actions': obs['last_actions'],
            'cargo_slots_available': obs['cargo_slots_available'],
            'home_race_requested': obs['home_race_requested'],
            'away_race_requested': obs['away_race_requested'],
            'action_result': obs['action_result'],
            'alerts': obs['alerts'],
            'game_loop': obs['game_loop'],
            'player': obs['player'],
            'control_groups': obs['control_groups'],
            'upgrades': obs['upgrades'],
            'available_actions_one_hot': obs['available_actions_one_hot']
        }, batch_size=1)

        time3 = time()

        for i in range(1):
            print()
            print(i)
            print(network_output[i])

        queued_logits = network_output[1].flatten()
        selected_unit_logits = network_output[2].flatten()
        shifted_logits4 = network_output[3].flatten()
        shifted_logits5 = network_output[4].flatten()
        control_group_id = network_output[5].flatten()
        minimap_logits = network_output[6].reshape(128, 128)
        screen1_logits = network_output[7].reshape(128, 128)
        screen2_logits = network_output[8].reshape(128, 128)

        action_type_logits = np.zeros(573)
        action_type_logits[action] = maxValue

        actionID = self.reverse_list[action]

        # Queue(2, )
        # Selected_Units(MAX_UNITS, )
        # Target_Unit(4, ), (5,), (10,)
        # Target_Point(128, 128), (128, 128), (128, 128)

        if actionID == 0:  # [(4,), (MAX_UNITS,)]
            shifted_logits4 = np.zeros(4)
            selected_unit_logits = np.zeros(MAX_UNITS)
            shifted_logits4[arg[0][0]] = maxValue
            selected_unit_logits[arg[1][0]] = maxValue
        elif actionID == 1:  # [(128, 128)]
            minimap_logits = np.zeros((128, 128))
            minimap_logits[arg[0][1]][arg[0][0]] = maxValue
        elif actionID == 2:  # [(2,), (128, 128)]
            queued_logits = np.zeros(2)
            queued_logits[arg[0][0]] = maxValue
            screen1_logits = np.zeros((128, 128))
            screen1_logits[arg[1][1]][arg[1][0]] = maxValue
        elif actionID == 3:  # [(2,), (128, 128)]
            queued_logits = np.zeros(2)
            queued_logits[arg[0][0]] = maxValue
            minimap_logits = np.zeros((128, 128))
            minimap_logits[arg[1][1]][arg[1][0]] = maxValue
        elif actionID == 4:  # [(4,)]
            shifted_logits4 = np.zeros(4)
            shifted_logits4[arg[0][0]] = maxValue
        if actionID == 5:  # [(2,)]
            queued_logits = np.zeros(2)
            queued_logits[arg[0][0]] = maxValue
        elif actionID == 6:  # [(4,), (128, 128)]
            shifted_logits4 = np.zeros(4)
            shifted_logits4[arg[0][0]] = maxValue
            screen1_logits = np.zeros((128, 128))
            screen1_logits[arg[1][1]][arg[1][0]] = maxValue
        elif actionID == 7:  # [(MAX_UNITS,)]
            selected_unit_logits = np.zeros(MAX_UNITS)
            selected_unit_logits[arg[0][0]] = maxValue
        elif actionID == 8:  # []
            pass
        elif actionID == 9:  # [(5,), (10,)]
            shifted_logits5 = np.zeros(5)
            control_group_id = np.zeros(10)
            shifted_logits5[arg[0][0]] = maxValue
            control_group_id[arg[1][0]] = maxValue
        elif actionID == 10:  # [(2,), (128, 128), (128, 128)]
            queued_logits = np.zeros(2)
            queued_logits[arg[0][0]] = maxValue
            screen1_logits = np.zeros((128, 128))
            screen1_logits[arg[1][1]][arg[1][0]] = maxValue
            screen2_logits = np.zeros((128, 128))
            screen2_logits[arg[2][1]][arg[2][0]] = maxValue
        elif actionID == 11:  # [(10,)]
            control_group_id = np.zeros(10)
            control_group_id[arg[0][0]] = maxValue

        time4 = time()

        self.startcraft_model.fit(
            x={
                'entity_list': obs['feature_units'],
                'minimap': obs['feature_minimap'],
                'single_select': obs['single_select'],
                'multi_select': obs['multi_select'],
                'build_queue': obs['build_queue'],
                'cargo': obs['cargo'],
                'production_queue': obs['production_queue'],
                'last_actions': obs['last_actions'],
                'cargo_slots_available': obs['cargo_slots_available'],
                'home_race_requested': obs['home_race_requested'],
                'away_race_requested': obs['away_race_requested'],
                'action_result': obs['action_result'],
                'alerts': obs['alerts'],
                'game_loop': obs['game_loop'],
                'player': obs['player'],
                'control_groups': obs['control_groups'],
                'upgrades': obs['upgrades'],
                'available_actions_one_hot': obs['available_actions_one_hot']
            },
            y={
                'action_type_logits': np.expand_dims(action_type_logits, axis=0),
                'queued_logits': np.expand_dims(queued_logits, axis=0),
                'selected_unit_logits': np.expand_dims(selected_unit_logits, axis=0),
                'shifted_logits4': np.expand_dims(shifted_logits4, axis=0),
                'shifted_logits5': np.expand_dims(shifted_logits5, axis=0),
                'control_group_id': np.expand_dims(control_group_id, axis=0),
                'minimap_logits': np.expand_dims(minimap_logits, axis=0),
                'screen1_logits': np.expand_dims(screen1_logits, axis=0),
                'screen2_logits': np.expand_dims(screen2_logits, axis=0)
            }
        )
        time5 = time()

        print("predict 이전 전처리", time1-time2, "predict", time2 - time3, "predict~update 전처리", time3-time4, "update", time4-time5)

    def model_predict(self, obs, available_action):
        # model.py로 옮겨서 사용할 model_predict
        self.delay_count += 1
        if self.delay_count != 7:
            return 0, []
        self.delay_count = 0

        for key in total_obs:
            obs[key] = np.asarray(obs[key].reshape(reshape_map[key])) / 5000.0

        network_output = self.startcraft_model.predict(x={
            'entity_list': obs['feature_units'],
            'minimap': obs['feature_minimap'],
            'single_select': obs['single_select'],
            'multi_select': obs['multi_select'],
            'build_queue': obs['build_queue'],
            'cargo': obs['cargo'],
            'production_queue': obs['production_queue'],
            'last_actions': obs['last_actions'],
            'cargo_slots_available': obs['cargo_slots_available'],
            'home_race_requested': obs['home_race_requested'],
            'away_race_requested': obs['away_race_requested'],
            'action_result': obs['action_result'],
            'alerts': obs['alerts'],
            'game_loop': obs['game_loop'],
            'player': obs['player'],
            'control_groups': obs['control_groups'],
            'upgrades': obs['upgrades'],
            'available_actions_one_hot': obs['available_actions_one_hot']
        }, batch_size=1)

        # 모든 네트워크 출력
        Action_Type = network_output[0].flatten().tolist()
        Queue = network_output[1].flatten().tolist()  # size 2
        Selected_Units = network_output[2].flatten().tolist()  # size MAX_UNITS
        Target_Unit = network_output[3].flatten().tolist() + network_output[4].flatten().tolist() + network_output[
            5].flatten().tolist()  # size 19
        # Target_Point = network_output[6].flatten().tolist() + network_output[7].flatten().tolist() + network_output[
        #     8].flatten().tolist()  # size 592

        network_output[6] = network_output[6].reshape(128, 128)
        network_output[7] = network_output[7].reshape(128, 128)
        network_output[8] = network_output[8].reshape(128, 128)

        actionlist = [Action_Type[i] for i in available_action]
        print("A", actionlist, available_action, np.argmax(actionlist))

        action = available_action[np.argmax(actionlist)]
        actionID = self.reverse_list[action]

        # Queue(2, )
        # Selected_Units(MAX_UNITS, )
        # Target_Unit(4, ), (5,), (10,)
        # Target_Point(128, 128), (84, 84), (84, 84)

        if actionID == 0:  # [(4,), (MAX_UNITS,)]
            arg1 = np.argmax(Target_Unit[:4])
            arg2 = np.argmax(Selected_Units)
            arg = [(arg1,), (arg2,)]

        elif actionID == 1:  # [(128, 128)], minimap : 6
            arg1 = np.unravel_index(network_output[6].argmax(), network_output[6].shape)
            arg = [arg1]

        elif actionID == 2:  # [(2,), (128, 128)], screen1 : 7
            arg1 = np.argmax(Queue)
            arg2 = np.unravel_index(network_output[7].argmax(), network_output[7].shape)
            arg = [(arg1,), arg2]

        elif actionID == 3:  # [(2,), (128, 128)], minimap : 6
            arg1 = np.argmax(Queue)
            arg2 = np.unravel_index(network_output[6].argmax(), network_output[6].shape)
            arg = [(arg1,), arg2]

        elif actionID == 4:  # [(4,)]
            arg1 = np.argmax(Target_Unit[:4])
            arg = [(arg1,)]

        elif actionID == 5:  # [(2,)]
            arg1 = np.argmax(Queue)
            arg = [(arg1,)]

        elif actionID == 6:  # [(4,), (128, 128)], screen2 : 7
            arg1 = np.argmax(Target_Unit[:4])
            arg2 = np.unravel_index(network_output[7].argmax(), network_output[7].shape)
            arg = [(arg1,), arg2]

        elif actionID == 7:  # [(MAX_UNITS,)]
            arg1 = np.argmax(Selected_Units)
            arg = [(arg1,)]

        elif actionID == 8:  # []
            arg = []

        elif actionID == 9:  # [(5,), (10,)]
            arg1 = np.argmax(Target_Unit[4:9])
            arg2 = np.argmax(Target_Unit[9:19])
            arg = [(arg1,), (arg2,)]

        elif actionID == 10:  # [(2,), (128, 128), (128, 128)], screen1 : 7, screen2 : 8
            arg1 = np.argmax(Queue)
            arg2 = np.unravel_index(network_output[7].argmax(), network_output[7].shape)
            arg3 = np.unravel_index(network_output[8].argmax(), network_output[8].shape)
            arg = [(arg1,), arg2, arg3]

        elif actionID == 11:  # [(10,)]
            arg1 = np.argmax(Target_Unit[9:19])
            arg = [(arg1,)]

        else:
            arg = 0, []

        return action, arg

    def step(self, obs):
        super(ModelAgent, self).step(obs)
        env = obs_pre_processing(obs.observation)

        for key in total_obs:
            env[key] = np.asarray(env[key].reshape(reshape_map[key]))

        function_id, args = self.model_predict(env, obs.observation.available_actions)
        print(function_id, ' ', args)
        # function_id, args = self.train_network(input_data, obs.observation.available_actions)
        # function_id = select_function(obs.observation.available_actions)
        # function_id = 1
        args = [[select_args(size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
