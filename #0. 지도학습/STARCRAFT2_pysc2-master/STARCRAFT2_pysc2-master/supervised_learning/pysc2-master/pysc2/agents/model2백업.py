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

from pysc2.agents import base_agent
from pysc2.lib import actions

from time import time

from pysc2.agents.model import DQN
import tensorflow as tf


def select_function(available_actions):
    # print("available action :", available_actions)
    return np.random.choice(available_actions)


def select_args(size):
    return np.random.randint(0, size)


MAX_POPULATION = 200
MAX_BUILD = 20
MAX_CARGO = 20
MAX_PRODUCTION = 5
max_len_mapping = {'multi_select': MAX_POPULATION,
                   'build_queue': MAX_BUILD,
                   'cargo': MAX_CARGO,
                   'production_queue': MAX_PRODUCTION}

key_test = {
    "single_select": (1, 7),
    "multi_select": (200, 7),
    "build_queue": (20, 7),
    "cargo": (20, 7),
    "production_queue": (5, 2),
    "last_actions": (1,),
    "cargo_slots_available": (1,),
    "home_race_requested": (1,),
    "away_race_requested": (1,),
    "feature_screen": (27, 84, 84),
    "feature_minimap": (11, 64, 64),
    "action_result": (1,),
    "alerts": (2,),
    "game_loop": (1,),
    "score_cumulative": (13,),
    "score_by_category": (11, 5),
    "score_by_vital": (3, 3),
    "player": (11,),
    "control_groups": (10, 2),
    "upgrades": (0,)}

reshape_map = {"single_select": (7,),
               "multi_select": (1400,),
               "build_queue": (140,),
               "cargo": (140,),
               "production_queue": (10,),
               "last_actions": (1,),
               "cargo_slots_available": (1,),
               "home_race_requested": (1,),
               "away_race_requested": (1,),
               "feature_screen": (190512,),
               "feature_minimap": (45056,),
               "action_result": (1,),
               "alerts": (2,),
               "game_loop": (1,),
               "score_cumulative": (13,),
               "score_by_category": (55,),
               "score_by_vital": (9,),
               "player": (11,),
               "control_groups": (20,),
               "upgrades": (1,),
               "available_actions_one_hot": (573,)}


def obs_pre_processing(observation):
    def pad_2d(observation, name):
        padding = max_len_mapping[name] - len(observation[name])
        if padding < 0:
            observation[name] = observation[name][:max_len_mapping[name]]
        else:
            observation[name] = np.pad(observation[name], ((0, padding), (0, 0)))

    def pad_1d(observation, name, max_len):
        padding = max_len - observation[name].size
        if padding < 0:
            observation[name] = observation[name][:max_len]
        else:
            observation[name] = np.pad(observation[name], (0, padding))

    if observation['single_select'].shape == (0, 7):  # single_select : (1, 7) 고정
        observation['single_select'] = np.pad(observation['single_select'], ((0, 1), (0, 0)))
    pad_2d(observation, 'multi_select')  # multi_select : (200, 7) 고정
    pad_2d(observation, 'build_queue')  # build_queue : (20, 7) 고정
    pad_2d(observation, 'cargo')  # cargo (20, 7) 고정
    pad_2d(observation, 'production_queue')  # production_queue (5, 2) 고정

    pad_1d(observation, 'last_actions', 1)  # last_actions (1,) 고정
    pad_1d(observation, 'alerts', 2)  # alerts
    pad_1d(observation, 'action_result', 1)  # action_result
    pad_1d(observation, 'upgrades', 1)  # upgrades

    observation['available_actions_one_hot'] = np.array(
        [0 for _ in range(573)], dtype=np.int32)
    for i in observation["available_actions"]:
        observation['available_actions_one_hot'][i] = 1

    input_data = [_ for _ in range(237955)]
    i = 0
    for key, reshape in reshape_map.items():
        for value in observation[key].reshape(reshape):
            input_data[i] = value
            i += 1

    return input_data


class ModelAgent(base_agent.BaseAgent):

    def arguments_processing(self, func_id, args_Qs):
        data_2 = np.argmax(args_Qs[0][0:2])
        data_4 = np.argmax(args_Qs[0][2:6])
        data_5 = np.argmax(args_Qs[0][6:11])
        data_10 = np.argmax(args_Qs[0][11:21])
        data_500 = np.argmax(args_Qs[0][21:521])
        data_84_1_0 = np.argmax(args_Qs[0][521:605])
        data_84_1_1 = np.argmax(args_Qs[0][605:689])
        data_84_2_0 = np.argmax(args_Qs[0][689:773])
        data_84_2_1 = np.argmax(args_Qs[0][773:857])
        data_64_0 = np.argmax(args_Qs[0][857:921])
        data_64_1 = np.argmax(args_Qs[0][921:985])

        data_dic = {
            (2,): data_2,
            (4,): data_4,
            (5,): data_5,
            (10,): data_10,
            (500,): data_500,
            (84, 84): [[data_84_1_0, data_84_1_1], [data_84_2_0, data_84_2_1]],
            (64, 64): [data_64_0, data_64_1]
        }

        flag_84 = False
        arguments = []
        for arg in self.action_spec.functions[func_id].args:
            if arg.sizes == (84, 84):
                if not flag_84:
                    arguments.append(data_dic[arg.sizes][0])
                    flag_84 = True
                else:
                    arguments.append(data_dic[arg.sizes][1])
            else:
                if arg.sizes == (64, 64):
                    arguments.append(data_dic[arg.sizes])
                else:
                    arguments.append([data_dic[arg.sizes]])

        return arguments

    """A random agent for starcraft."""

    def train_network(self, input_data, available_actions):

        func_input_data = np.array(input_data).reshape(1, len(input_data))
        full_Qs = self.full_network.predict(func_input_data)
        # Qs = self.function_network.predict(func_input_data)
        func_Qs = full_Qs[0][:573]
        functions = func_Qs
        for i in range(len(functions)):
            if i not in available_actions:
                functions[i] = 0
        args_Qs = [full_Qs[0][573:]]
        func_id = np.argmax(func_Qs)
        arguments = self.arguments_processing(func_id, args_Qs)

        return func_id, arguments

    def step(self, obs):
        super(ModelAgent, self).step(obs)
        input_data = obs_pre_processing(obs.observation)
        print("input_data", len(input_data))

        input_data = []
        for key, value in obs.observation.items():
            if key != 'map_name':
                tmp = list(obs.observation[key].flatten())
                while len(tmp) < 100:
                    tmp.append(0)
                input_data.extend(tmp)

        print(len(input_data))
        epsilon = 0.1
        if np.random.random() < epsilon:
            # numpy.random.randint : 이미 있는 데이터 집합에서 일부를 무작위로 선택(sampling)
            function_id = select_function(obs.observation.available_actions)
            args = [[select_args(size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
            # print("랜덤 :", "func :", function_id, "args :", args)
        else:
            function_id, args = self.train_network(input_data, set(obs.observation.available_actions))
            # print("네트워크 :", "func :", function_id, "args :", args)

        print(function_id, args)
        return actions.FunctionCall(function_id, args)

    def update(self, obs, func, args):
        st = time()
        input_data = []
        for key, value in obs.items():
            if key != 'map_name':
                tmp = list(obs[key].flatten())
                while len(tmp) < 100:
                    tmp.append(0)
                input_data.extend(tmp)
        mi = time()
        output_data = [0 for _ in range(1558)]
        output_data[func] = 1

        argID = self.reverse_list[func]

        action_index = 573
        index_2 = action_index
        index_4 = action_index + 2
        index_5 = action_index + 6
        index_10 = action_index + 11
        index_500 = action_index + 21
        index_84_1_0 = action_index + 521
        index_84_1_1 = action_index + 605
        index_84_2_0 = action_index + 689
        index_84_2_1 = action_index + 773
        index_64_0 = action_index + 857
        index_64_1 = action_index + 921

        if argID == 0:
            arg0 = args[0][0]
            arg1 = args[1][0]
            output_data[index_4 + arg0] = 1
            output_data[index_500 + arg1] = 1
        elif argID == 1:
            arg0 = args[0][0]
            arg1 = args[0][1]
            output_data[index_64_0 + arg0] = 1
            output_data[index_64_1 + arg1] = 1
        elif argID == 2:
            arg0 = args[0][0]
            arg1 = args[1][0]
            arg2 = args[1][1]
            output_data[index_2 + arg0] = 1
            output_data[index_84_1_0 + arg1] = 1
            output_data[index_84_1_1 + arg2] = 1
        elif argID == 3:
            arg0 = args[0][0]
            arg1 = args[1][0]
            arg2 = args[1][1]
            output_data[index_2 + arg0] = 1
            output_data[index_64_0 + arg1] = 1
            output_data[index_64_1 + arg2] = 1
        elif argID == 4:
            arg0 = args[0][0]
            output_data[index_4 + arg0] = 1
        elif argID == 5:
            arg0 = args[0][0]
            output_data[index_2 + arg0] = 1
        elif argID == 6:
            arg0 = args[0][0]
            arg1 = args[1][0]
            arg2 = args[1][1]
            output_data[index_4 + arg0] = 1
            output_data[index_84_1_0 + arg1] = 1
            output_data[index_84_1_1 + arg2] = 1
        elif argID == 7:
            arg0 = args[0][0]
            output_data[index_500 + arg0] = 1
        elif argID == 8:
            pass
        elif argID == 9:
            arg0 = args[0][0]
            arg1 = args[1][0]
            output_data[index_5 + arg0] = 1
            output_data[index_10 + arg1] = 1
        elif argID == 10:
            arg0 = args[0][0]
            arg1 = args[1][0]
            arg2 = args[1][1]
            arg3 = args[2][0]
            arg4 = args[2][1]
            output_data[index_2 + arg0] = 1
            output_data[index_84_1_0 + arg1] = 1
            output_data[index_84_1_1 + arg2] = 1
            output_data[index_84_2_0 + arg3] = 1
            output_data[index_84_2_1 + arg4] = 1
        elif argID == 11:
            arg0 = args[0][0]
            output_data[index_10 + arg0] = 1
        mi2 = time()
        self.full_network.update(np.reshape(input_data, [1, self.full_network.input_size]), np.array(output_data))
        en = time()
        # print(st, mi, mi2, en)
        print(mi - st, mi2 - mi, en - mi2)