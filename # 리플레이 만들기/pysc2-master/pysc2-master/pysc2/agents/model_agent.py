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

def obs_pre_processing(observation):
    def pad_2d(observation, name):
        padding = max_len_mapping[name] - len(observation[name])
        if padding <= 0:
            observation[name] = observation[name][:max_len_mapping[name]]
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

LSTM1_logs = keras.callbacks.TensorBoard('./tensorboard_logs/LSTM1_logs',
                                                   update_freq=1000)


class ModelAgent(base_agent.BaseAgent):
    def select_obs(self, obs):
        obs = obs_pre_processing(obs)
        for key in total_obs:
            obs[key] = np.asarray(obs[key].reshape(reshape_map[key]))

        return {'feature_screen': np.array(obs['feature_screen'].tolist()), 'feature_minimap': obs['feature_minimap'], 'player': obs['player'], 'game_loop': obs['game_loop']}

    def small_update_play(self, action, obs):
        maxValue = 1

        obs = obs_pre_processing(obs)
        for key in total_obs:
            obs[key] = np.asarray(obs[key].reshape(reshape_map[key]))

        action_type_logits = np.zeros(573)
        action_type_logits[action] = maxValue

        obs['feature_units'] = normalize(obs['feature_units'])
        obs['player'] = normalize(obs['player'])

        obs['game_loop'] = obs['game_loop'] / (22.4 * 1200)

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

    def step(self, obs):
        super(ModelAgent, self).step(obs)
        env = obs_pre_processing(obs.observation)
        for key in total_obs:
            env[key] = np.asarray(env[key].reshape(reshape_map[key]))

        function_id, args = self.model_predict(env, obs.observation.available_actions)
        args = [[select_args(size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)