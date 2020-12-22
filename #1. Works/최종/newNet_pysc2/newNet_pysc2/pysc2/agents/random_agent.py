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
"feature_minimap" :(11, 64, 64),
"action_result": (1,),
"alerts": (2,),
"game_loop": (1,),
"score_cumulative": (13,),
"score_by_category": (11, 5),
"score_by_vital": (3, 3),
"player":(11,),
"control_groups": (10, 2),
"upgrades": (0,)}

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
  pad_2d(observation, 'multi_select')       # multi_select : (200, 7) 고정
  pad_2d(observation, 'build_queue')        # build_queue : (20, 7) 고정
  pad_2d(observation, 'cargo')              # cargo (20, 7) 고정
  pad_2d(observation, 'production_queue')   # production_queue (5, 2) 고정

  pad_1d(observation, 'last_actions', 1)    # last_actions (1,) 고정
  pad_1d(observation, 'alerts', 2)          # alerts
  pad_1d(observation, 'action_result', 1)   # action_result

  # 길이가 고정되었는지 테스트
  # for key in key_test.keys():
  #   if observation[key].shape != key_test[key]:
  #     print(key, observation[key].shape, observation[key])

  observation['available_actions_one_hot'] = np.array(
      [0 for _ in range(573)], dtype=np.int32)
  for i in observation["available_actions"]:
    observation['available_actions_one_hot'][i] = 1

  # print(observation['available_actions'], observation['available_actions_one_hot'])


class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    super(RandomAgent, self).step(obs)

    ACTION = 5
    ARGS = [(0,), (0,)]

    function_id = 0

    # print(obs.observation.available_actions)

    for act in obs.observation.available_actions:
      if act == ACTION:
        self.count = self.count + 1
        function_id = 0
        if self.count % 10 == 0:
          print(self.count)
        if self.count == 100:
          self.count = 0
          function_id = act
        break

    if function_id == ACTION:
      args = ARGS
    else:
      args = [[np.random.randint(0, size) for size in arg.sizes]
              for arg in self.action_spec.functions[function_id].args]

    if function_id == ACTION:
      print(function_id, args)

    return actions.FunctionCall(function_id, args)