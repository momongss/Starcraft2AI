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

from pysc2.agents.Network.architecture import get_action_Model, get_point_Model
from pysc2.agents import base_agent
from pysc2.lib import actions

import sys

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


class ModelAgent(base_agent.BaseAgent):
    """A model agent for starcraft."""

    def __init__(self):
        super(ModelAgent, self).__init__()
        self.action_model = get_action_Model()
        self.point_model = get_point_Model()

        try:
            self.action_model.load_weights('./action_checkpoints/good_checkpoint')
            self.point_model.load_weights('./point_checkpoints/good_checkpoint')
            print('Success loading')
        except:
            print('Fail loading')

        # self.action_model.save_weights('./action_checkpoints/good_checkpoint')
        # self.point_model.save_weights('./point_checkpoints/good_checkpoint')

        self.DISCOUNT = 0.9
        self.attack = 0
        self.valid_actions = {}

        self.prev_state = 0
        self.train_count = 0

        self.available_count = 0
        self.AVAILABLE_DELAY = 112

        self.epsilon = 0.3

        self.dead_scv = [1, 0, 0, 0]
        self.attack_point_list = [
            [96, 127],
            [127, 102],
            [127, 94],
            [127, 98],
            [127, 127],
            [82, 127],
            [127, 81]
        ]

    def action_predict(self, obs):
        return self.action_model.predict(x=self.state2input(obs))

    def attack_point_predict(self, obs):
        return self.point_model.predict(x=self.state2input(obs))

    def state2input(self, obs):
        return {'feature_minimap': np.array(list(obs.observation['feature_minimap'])).reshape(1, 128, 128, 11),
                'player': np.array(list(obs.observation['player'])).reshape(1, 11),
                'game_loop': np.array(list(obs.observation['game_loop'])).reshape(1, 1)}

    def train_action_Q(self, next_state, reward, done, state):
        Qf = self.action_model.predict(x=self.state2input(state))
        if done:
            Qf[0][self.special_action] = reward
        else:
            next_Qf = self.action_model.predict(x=self.state2input(next_state))
            Qf[0][self.special_action] = reward + self.DISCOUNT * np.max(next_Qf[0])
        print('action id:', self.special_action, 'Q:', Qf)
        self.action_model.fit(x=self.state2input(state), y=Qf)
        return

    def train_point_Q(self, next_state, reward, done, state):
        Qp = self.point_model.predict(x=self.state2input(state))
        if done:
            Qp[0][self.attack] = reward
        else:
            next_Qp = self.point_model.predict(x=self.state2input(next_state))
            Qp[0][self.attack] = reward + self.DISCOUNT * np.max(next_Qp[0])
        print('point id:', self.attack, 'Q:', Qp)
        self.point_model.fit(x=self.state2input(state), y=Qp)
        return

    def step(self, obs):
        super(ModelAgent, self).step(obs)
        # 딜레이 적용
        if self.action_delay != 0:
            self.action_delay -= 1
            return actions.FunctionCall(0, [])

        gameloop = obs.observation['game_loop'][0]

        function_id = 0
        args = []

        # 일정스텝동안 고정된 동작
        if gameloop < self.copystep:
            if self.copylist.get(gameloop):
                function_id = self.copylist[gameloop][0]
                if function_id not in obs.observation.available_actions:
                    if function_id != 477:
                        # sys.exit()
                        pass
                    return actions.FunctionCall(0, [])
                args = self.copylist[gameloop][1]
                # print(gameloop, function_id, args)

        # 네트워크에 의해 결정되는 동작
        else:
            if self.train_count > 0:
                # self.train_action_Q(obs, obs.reward, obs.last(), self.prev_state)
                # self.train_point_Q(obs, obs.reward, obs.last(), self.prev_state)
                pass

            self.train_count += 1
            self.prev_state = obs
            # 진행중인 action이 없는 경우
            if self.small_step_count == 0:
                # predict으로 액션을 정함
                action_logits = self.action_predict(obs)
                if np.random.random() > self.epsilon:
                    self.special_action = np.argmax(action_logits[0])

                    attack_point_logits = self.attack_point_predict(obs)
                    self.attack = np.argmax(attack_point_logits[0])
                    self.small_step_count = 2

                    self.attack_point = self.attack_point_list[self.attack]
                else:
                    self.special_action = np.random.randint(0, 3)
                    self.attack = np.random.randint(0, 7)
                    self.attack_point = self.attack_point_list[self.attack]
                    self.small_step_count = 2

            # 진행중인 action이 존재하는 경우
            print("special", self.special_action, self.small_step_count)
            if self.small_step_count != 0:
                if self.special_action == self.ACTION_TRAIN_MARIN:
                    if self.small_step_count == 2:
                        # 8번부대 선택
                        function_id = 4
                        if function_id not in obs.observation.available_actions:
                            self.small_step_count = 0
                            return actions.FunctionCall(0, [])
                        args = [[0], [8]]
                        self.small_step_count -= 1

                    elif self.small_step_count == 1:
                        # 마린생산
                        function_id = 477
                        if function_id not in obs.observation.available_actions:
                            print(self.available_count)
                            if self.available_count != self.AVAILABLE_DELAY:
                                self.available_count += 1
                            else:
                                self.small_step_count = 0
                            return actions.FunctionCall(0, [])
                        args = [[0]]
                        self.small_step_count -= 1
                        self.action_delay = self.DELAY_TIME

                elif self.special_action == self.ACTION_BUILD_SUPPLY:
                    if self.small_step_count == 2:
                        # 1, 2, 3번 부대 중 하나 선택
                        function_id = 4
                        if function_id not in obs.observation.available_actions:
                            if self.dead_scv[1] and self.dead_scv[2] and self.dead_scv[3]:
                                self.small_step_count = 0
                            elif self.dead_scv[self.last_used_scv] == 1:
                                self.last_used_scv = (self.last_used_scv % 3) + 1
                            return actions.FunctionCall(0, [])
                        self.last_used_scv = (self.last_used_scv % 3) + 1
                        args = [[0], [self.last_used_scv]]
                        self.small_step_count -= 1

                    elif self.small_step_count == 1:
                        # 서플짓기
                        function_id = 91
                        if function_id not in obs.observation.available_actions:
                            print(self.available_count)
                            if self.available_count != self.AVAILABLE_DELAY:
                                self.available_count += 1
                            else:
                                self.small_step_count = 0
                            return actions.FunctionCall(0, [])
                        args = [[0], self.supply_point[self.supply_count]]
                        if self.supply_count != 7:
                            self.supply_count += 1
                        self.small_step_count -= 1
                        self.action_delay = self.DELAY_TIME

                elif self.special_action == self.ACTION_ATTACK_MINIMAP:
                    if self.small_step_count == 2:
                        # 모든 마린 선택
                        function_id = 7
                        if function_id not in obs.observation.available_actions:
                            self.small_step_count = 0
                            return actions.FunctionCall(0, [])
                        args = [[0]]
                        self.small_step_count -= 1

                    elif self.small_step_count == 1:
                        # 공격 predict 한 시점에 미리 정해진 좌표 사용
                        function_id = 13
                        if function_id not in obs.observation.available_actions:
                            print(self.available_count)
                            if self.available_count != self.AVAILABLE_DELAY:
                                self.available_count += 1
                            else:
                                self.small_step_count = 0
                            return actions.FunctionCall(0, [])
                        args = [[0], self.attack_point]
                        self.small_step_count -= 1
                        self.action_delay = self.DELAY_TIME

        if function_id != 0:
            self.available_count = 0
        return actions.FunctionCall(function_id, args)