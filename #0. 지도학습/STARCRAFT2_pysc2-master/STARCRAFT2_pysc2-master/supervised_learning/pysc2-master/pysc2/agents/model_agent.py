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
    'feature_units': (1, 500, 46),
    'feature_minimap': (1, 128, 128, 11),
    'single_select': (1, 7),
    'multi_select': (1, 1400),
    'build_queue': (1, 140),
    'cargo': (1, 140),
    'production_queue': (1, 10),
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

MAX_UNITS = 500
MAX_POPULATION = 200
MAX_BUILD = 20
MAX_CARGO = 20
MAX_PRODUCTION = 5
max_len_mapping = {'multi_select': MAX_POPULATION,
                   'build_queue': MAX_BUILD,
                   'cargo': MAX_CARGO,
                   'production_queue': MAX_PRODUCTION,
                   'feature_units': MAX_UNITS}



def obs_pre_processing(observation):
  def pad_2d(observation, name):
    padding = max_len_mapping[name] - len(observation[name])
    if padding <= 0:
      observation[name] = observation[name][:max_len_mapping[name]]
    # elif padding < max_len_mapping[name]:
    else:
      observation[name] = np.pad(observation[name], ((0, padding), (0, 0)))
    # else:
      # observation[name] = np.zeros((max_len_mapping[name], len(observation[name])))

  def pad_4_units(observation, name):
      padding = max_len_mapping[name] - len(observation[name])
      if padding <= 0:
          observation[name] = observation[name][:max_len_mapping[name]]
      elif padding < 500:
          observation[name] = np.pad(observation[name], ((0, padding), (0, 0)))
      else:
          observation[name] = np.zeros((500, 46))

  def pad_1d(observation, name, max_len):
    padding = max_len - observation[name].size
    if padding < 0:
      observation[name] = observation[name][:max_len]
    else:
      observation[name] = np.pad(observation[name], (0, padding))

  if observation['single_select'].shape == (0, 7):  # single_select : (1, 7) 고정
    observation['single_select'] = np.pad(observation['single_select'], ((0, 1), (0, 0)))
  pad_4_units(observation, 'feature_units')
  pad_2d(observation, 'multi_select')       # multi_select : (200, 7) 고정
  pad_2d(observation, 'build_queue')        # build_queue : (20, 7) 고정
  pad_2d(observation, 'cargo')              # cargo (20, 7) 고정
  pad_2d(observation, 'production_queue')   # production_queue (5, 2) 고정

  pad_1d(observation, 'last_actions', 1)    # last_actions (1,) 고정
  pad_1d(observation, 'alerts', 2)          # alerts
  pad_1d(observation, 'action_result', 1)   # action_result
  pad_1d(observation, 'upgrades', 1)        # upgrades

  observation['available_actions_one_hot'] = np.array(
      [0 for _ in range(573)], dtype=np.int32)
  for i in observation["available_actions"]:
      observation['available_actions_one_hot'][i] = 1

  return observation

epsilon = 0.1


class ModelAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(ModelAgent, self).step(obs)
    env = obs_pre_processing(obs.observation)

    for i, key in enumerate(total_obs):
        env[key] = np.asarray(env[key].reshape(reshape_map[key]))

    function_id, args = self.model_predict(obs.observation.available_actions, env)
    self.model_update(function_id, args, env)
    # function_id, args = self.train_network(input_data, obs.observation.available_actions)
    # function_id = select_function(obs.observation.available_actions)
    # args = [[select_args(size) for size in arg.sizes]
    #             for arg in self.action_spec.functions[function_id].args]
    # print('func', function_id, 'args', args)
    return actions.FunctionCall(function_id, args)

  def model_update_play(self, action, arg, obs):
      maxValue = 1
      minValue = 0

      obs = obs_pre_processing(obs.observation)

      network_output = self.startcraft_model.predict(x={
          'entity_list': np.expand_dims(obs['feature_units'], axis=0),
          'minimap': np.expand_dims(obs['feature_minimap'], axis=0),
          'single_select': np.expand_dims(obs['single_select'], axis=0),
          'multi_select': np.expand_dims(obs['multi_select'], axis=0),
          'build_queue': np.expand_dims(obs['build_queue'], axis=0),
          'cargo': np.expand_dims(obs['cargo'], axis=0),
          'production_queue': np.expand_dims(obs['production_queue'], axis=0),
          'last_actions': np.expand_dims(obs['last_actions'], axis=0),
          'cargo_slots_available': np.expand_dims(obs['cargo_slots_available'], axis=0),
          'home_race_requested': np.expand_dims(obs['home_race_requested'], axis=0),
          'away_race_requested': np.expand_dims(obs['away_race_requested'], axis=0),
          'action_result': np.expand_dims(obs['action_result'], axis=0),
          'alerts': np.expand_dims(obs['alerts'], axis=0),
          'game_loop': np.expand_dims(obs['game_loop'], axis=0),
          'score_cumulative': np.expand_dims(obs['score_cumulative'], axis=0),
          'score_by_category': np.expand_dims(obs['score_by_category'], axis=0),
          'score_by_vital': np.expand_dims(obs['score_by_vital'], axis=0),
          'player': np.expand_dims(obs['player'], axis=0),
          'control_groups': np.expand_dims(obs['control_groups'], axis=0),
          'upgrades': np.expand_dims(obs['upgrades'], axis=0),
          'available_actions_one_hot': np.expand_dims(obs['available_actions_one_hot'], axis=0)})

      # 모든 네트워크 출력
      Action_Type = network_output[0].flatten().tolist()
      Queue = network_output[1].flatten().tolist()  # size 2
      Selected_Units = network_output[2].flatten().tolist()  # size 500
      Target_Unit = network_output[3].flatten().tolist() + network_output[4].flatten().tolist() + network_output[5].flatten().tolist()  # size 19
      Target_Point = network_output[6].flatten().tolist() + network_output[7].flatten().tolist() + network_output[8].flatten().tolist()  # size 592

      Action_Type[action] = maxValue

      actionID = self.reverse_list[action]

      # Queue(2, )
      # Selected_Units(500, )
      # Target_Unit(4, ), (5,), (10,)
      # Target_Point(128, 128), (128, 128), (128, 128)

      if actionID == 0:  # [(4,), (500,)]
          Target_Unit[:4] = [minValue for _ in range(4)]
          Selected_Units = [minValue for _ in range(500)]
          Target_Unit[arg[0][0]] = maxValue
          Selected_Units[arg[1][0]] = maxValue
      if actionID == 1:  # [(128, 128)]
          Target_Point[:256] = [minValue for _ in range(256)]
          Target_Point[arg[0][0]] = maxValue
          Target_Point[128 + arg[0][1]] = maxValue
      if actionID == 2:  # [(2,), (128, 128)]
          Queue = [0 for _ in range(2)]
          Target_Point[256:512] = [minValue for _ in range(256)]
          Queue[arg[0][0]] = maxValue
          Target_Point[256 + arg[1][0]] = maxValue
          Target_Point[256 + 128 + arg[1][1]] = maxValue
      if actionID == 3:  # [(2,), (128, 128)]
          Queue = [0 for _ in range(2)]
          Target_Point[:256] = [minValue for _ in range(256)]
          Queue[arg[0][0]] = maxValue
          Target_Point[arg[1][0]] = maxValue
          Target_Point[128 + arg[1][1]] = maxValue
      if actionID == 4:  # [(4,)]
          Target_Unit[:4] = [minValue for _ in range(4)]
          Target_Unit[arg[0][0]] = maxValue
      if actionID == 5:  # [(2,)]
          Queue = [0 for _ in range(2)]
          Queue[arg[0][0]] = maxValue
      if actionID == 6:  # [(4,), (128, 128)]
          Target_Unit[:4] = [minValue for _ in range(4)]
          Target_Point[256:512] = [minValue for _ in range(256)]
          Target_Unit[arg[0][0]] = maxValue
          Target_Point[256 + arg[1][0]] = maxValue
          Target_Point[256 + 128 + arg[1][1]] = maxValue
      if actionID == 7:  # [(500,)]
          Selected_Units = [minValue for _ in range(500)]
          Selected_Units[arg[0][0]] = maxValue
      if actionID == 8:  # []
          pass
      if actionID == 9:  # [(5,), (10,)]
          Target_Unit[4:19] = [minValue for _ in range(15)]
          Target_Unit[4 + arg[0][0]] = maxValue
          Target_Unit[4 + 5 + arg[1][0]] = maxValue
      if actionID == 10:  # [(2,), (128, 128), (128, 128)]
          Queue = [0 for _ in range(2)]
          Target_Point[256:768] = [minValue for _ in range(512)]
          Queue[arg[0][0]] = maxValue
          Target_Point[256 + arg[1][0]] = maxValue
          Target_Point[256 + 128 + arg[1][1]] = maxValue
          Target_Point[256 + 128 + 128 + arg[2][0]] = maxValue
          Target_Point[256 + 128 + 128 + 128 + arg[2][1]] = maxValue
      if actionID == 11:  # [(10,)]
          Target_Unit[9:19] = [0 for _ in range(10)]
          Target_Unit[4 + 5 + arg[0][0]] = maxValue

      return

  def model_update(self, action, arg, obs):
      maxValue = 1
      minValue = 0

      obs = obs_pre_processing(obs.observation)

      for i, key in enumerate(total_obs):
          obs[key] = np.asarray(obs[key].reshape(reshape_map[key]))

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
          'score_cumulative': obs['score_cumulative'],
          'score_by_category': obs['score_by_category'],
          'score_by_vital': obs['score_by_vital'],
          'player': obs['player'],
          'control_groups': obs['control_groups'],
          'upgrades': obs['upgrades'],
          'available_actions_one_hot': obs['available_actions_one_hot']})

      # 모든 네트워크 출력
      Action_Type = network_output[0].flatten().tolist()
      Queue = network_output[1].flatten().tolist()  # size 2
      Selected_Units = network_output[2].flatten().tolist()  # size 500
      Target_Unit = network_output[3].flatten().tolist() + network_output[4].flatten().tolist() + network_output[5].flatten().tolist()  # size 19
      Target_Point = network_output[6].flatten().tolist() + network_output[7].flatten().tolist() + network_output[8].flatten().tolist()  # size 592

      Action_Type[action] = maxValue

      actionID = self.reverse_list[action]

      # Queue(2, )
      # Selected_Units(500, )
      # Target_Unit(4, ), (5,), (10,)
      # Target_Point(128, 128), (128, 128), (128, 128)

      if actionID == 0:  # [(4,), (500,)]
          Target_Unit[:4] = [minValue for _ in range(4)]
          Selected_Units = [minValue for _ in range(500)]
          Target_Unit[arg[0][0]] = maxValue
          Selected_Units[arg[1][0]] = maxValue
      if actionID == 1:  # [(128, 128)]
          Target_Point[:256] = [minValue for _ in range(256)]
          Target_Point[arg[0][0]] = maxValue
          Target_Point[128 + arg[0][1]] = maxValue
      if actionID == 2:  # [(2,), (128, 128)]
          Queue = [0 for _ in range(2)]
          Target_Point[256:512] = [minValue for _ in range(256)]
          Queue[arg[0][0]] = maxValue
          Target_Point[256 + arg[1][0]] = maxValue
          Target_Point[256 + 128 + arg[1][1]] = maxValue
      if actionID == 3:  # [(2,), (128, 128)]
          Queue = [0 for _ in range(2)]
          Target_Point[:256] = [minValue for _ in range(256)]
          Queue[arg[0][0]] = maxValue
          Target_Point[arg[1][0]] = maxValue
          Target_Point[128 + arg[1][1]] = maxValue
      if actionID == 4:  # [(4,)]
          Target_Unit[:4] = [minValue for _ in range(4)]
          Target_Unit[arg[0][0]] = maxValue
      if actionID == 5:  # [(2,)]
          Queue = [0 for _ in range(2)]
          Queue[arg[0][0]] = maxValue
      if actionID == 6:  # [(4,), (128, 128)]
          Target_Unit[:4] = [minValue for _ in range(4)]
          Target_Point[256:512] = [minValue for _ in range(256)]
          Target_Unit[arg[0][0]] = maxValue
          Target_Point[256 + arg[1][0]] = maxValue
          Target_Point[256 + 128 + arg[1][1]] = maxValue
      if actionID == 7:  # [(500,)]
          Selected_Units = [minValue for _ in range(500)]
          Selected_Units[arg[0][0]] = maxValue
      if actionID == 8:  # []
          pass
      if actionID == 9:  # [(5,), (10,)]
          Target_Unit[4:19] = [minValue for _ in range(15)]
          Target_Unit[4 + arg[0][0]] = maxValue
          Target_Unit[4 + 5 + arg[1][0]] = maxValue
      if actionID == 10:  # [(2,), (128, 128), (128, 128)]
          Queue = [0 for _ in range(2)]
          Target_Point[256:768] = [minValue for _ in range(512)]
          Queue[arg[0][0]] = maxValue
          Target_Point[256 + arg[1][0]] = maxValue
          Target_Point[256 + 128 + arg[1][1]] = maxValue
          Target_Point[256 + 128 + 128 + arg[2][0]] = maxValue
          Target_Point[256 + 128 + 128 + 128 + arg[2][1]] = maxValue
      if actionID == 11:  # [(10,)]
          Target_Unit[9:19] = [0 for _ in range(10)]
          Target_Unit[4 + 5 + arg[0][0]] = maxValue

      return

  def model_predict_play(self, available_action, obs):
      # model.py로 옮겨서 사용할 model_predict
      obs = obs_pre_processing(obs.observation)

      network_output = self.startcraft_model.predict(x={
          'entity_list': np.expand_dims(obs['feature_units'], axis=0),
          'minimap': np.expand_dims(obs['feature_minimap'],axis=0),
          'single_select': np.expand_dims(obs['single_select'], axis=0),
          'multi_select': np.expand_dims(obs['multi_select'], axis=0),
          'build_queue': np.expand_dims(obs['build_queue'], axis=0),
          'cargo': np.expand_dims(obs['cargo'], axis=0),
          'production_queue': np.expand_dims(obs['production_queue'], axis=0),
          'last_actions': np.expand_dims(obs['last_actions'], axis=0),
          'cargo_slots_available': np.expand_dims(obs['cargo_slots_available'], axis=0),
          'home_race_requested': np.expand_dims(obs['home_race_requested'], axis=0),
          'away_race_requested': np.expand_dims(obs['away_race_requested'], axis=0),
          'action_result': np.expand_dims(obs['action_result'], axis=0),
          'alerts': np.expand_dims(obs['alerts'], axis=0),
          'game_loop': np.expand_dims(obs['game_loop'], axis=0),
          'score_cumulative': np.expand_dims(obs['score_cumulative'], axis=0),
          'score_by_category': np.expand_dims(obs['score_by_category'], axis=0),
          'score_by_vital': np.expand_dims(obs['score_by_vital'], axis=0),
          'player': np.expand_dims(obs['player'], axis=0),
          'control_groups': np.expand_dims(obs['control_groups'], axis=0),
          'upgrades': np.expand_dims(obs['upgrades'], axis=0),
          'available_actions_one_hot': np.expand_dims(obs['available_actions_one_hot'], axis=0)})

      # 모든 네트워크 출력
      Action_Type = network_output[0].flatten().tolist()
      Queue = network_output[1].flatten().tolist()  # size 2
      Selected_Units = network_output[2].flatten().tolist()  # size 500
      Target_Unit = network_output[3].flatten().tolist() + network_output[4].flatten().tolist() + network_output[5].flatten().tolist()         # size 19
      Target_Point = network_output[6].flatten().tolist() + network_output[7].flatten().tolist() + network_output[8].flatten().tolist()        # size 592

      actionlist = [Action_Type[i] for i in available_action]

      action = available_action[np.argmax(actionlist)]
      actionID = self.reverse_list[action]

      # Queue(2, )
      # Selected_Units(500, )
      # Target_Unit(4, ), (5,), (10,)
      # Target_Point(128, 128), (84, 84), (84, 84)

      if actionID == 0:  # [(4,), (500,)]
          arg1 = np.argmax(Target_Unit[:4])
          arg2 = np.argmax(Selected_Units)
          arg = [(arg1,), (arg2,)]
          return action, arg
      if actionID == 1:  # [(128, 128)]
          arg1 = np.argmax(Target_Point[:128])
          arg2 = np.argmax(Target_Point[128:256])
          arg = [(arg1, arg2)]
          return action, arg
      if actionID == 2:  # [(2,), (128, 128)]
          arg1 = np.argmax(Queue)
          arg2 = np.argmax(Target_Point[256:384])
          arg3 = np.argmax(Target_Point[384:512])
          arg = [(arg1,), (arg2, arg3)]
          return action, arg
      if actionID == 3:  # [(2,), (128, 128)]
          arg1 = np.argmax(Queue)
          arg2 = np.argmax(Target_Point[:128])
          arg3 = np.argmax(Target_Point[128:256])
          arg = [(arg1,), (arg2, arg3)]
          return action, arg
      if actionID == 4:  # [(4,)]
          arg1 = np.argmax(Target_Unit[:4])
          arg = [(arg1,)]
          return action, arg
      if actionID == 5:  # [(2,)]
          arg1 = np.argmax(Queue)
          arg = [(arg1,)]
          return action, arg
      if actionID == 6:  # [(4,), (128, 128)]
          arg1 = np.argmax(Target_Unit[:4])
          arg2 = np.argmax(Target_Point[256:384])
          arg3 = np.argmax(Target_Point[384:512])
          arg = [(arg1,), (arg2, arg3)]
          return action, arg
      if actionID == 7:  # [(500,)]
          arg1 = np.argmax(Selected_Units)
          arg = [(arg1,)]
          return action, arg
      if actionID == 8:  # []
          arg = []
          return action, arg
      if actionID == 9:  # [(5,), (10,)]
          arg1 = np.argmax(Target_Unit[4:9])
          arg2 = np.argmax(Target_Unit[9:19])
          arg = [(arg1,), (arg2,)]
          return action, arg
      if actionID == 10:  # [(2,), (128, 128), (128, 128)]
          arg1 = np.argmax(Queue)
          arg2 = np.argmax(Target_Point[256:384])
          arg3 = np.argmax(Target_Point[384:512])
          arg4 = np.argmax(Target_Point[512:640])
          arg5 = np.argmax(Target_Point[640:768])
          arg = [(arg1,), (arg2, arg3), (arg4, arg5)]
          return action, arg
      if actionID == 11:  # [(10,)]
          arg1 = np.argmax(Target_Unit[9:19])
          arg = [(arg1,)]
          return action, arg

      return

  def model_predict(self, available_action, obs):
      # model.py로 옮겨서 사용할 model_predict
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
          'score_cumulative': obs['score_cumulative'],
          'score_by_category': obs['score_by_category'],
          'score_by_vital': obs['score_by_vital'],
          'player': obs['player'],
          'control_groups': obs['control_groups'],
          'upgrades': obs['upgrades'],
          'available_actions_one_hot': obs['available_actions_one_hot']})

      # 모든 네트워크 출력
      Action_Type = network_output[0].flatten().tolist()
      Queue = network_output[1].flatten().tolist()  # size 2
      Selected_Units = network_output[2].flatten().tolist()  # size 500
      Target_Unit = network_output[3].flatten().tolist() + network_output[4].flatten().tolist() + network_output[5].flatten().tolist()         # size 19
      Target_Point = network_output[6].flatten().tolist() + network_output[7].flatten().tolist() + network_output[8].flatten().tolist()        # size 592

      actionlist = [Action_Type[i] for i in available_action]

      action = available_action[np.argmax(actionlist)]
      actionID = self.reverse_list[action]

      # Queue(2, )
      # Selected_Units(500, )
      # Target_Unit(4, ), (5,), (10,)
      # Target_Point(128, 128), (84, 84), (84, 84)

      if actionID == 0:  # [(4,), (500,)]
          arg1 = np.argmax(Target_Unit[:4])
          arg2 = np.argmax(Selected_Units)
          arg = [(arg1,), (arg2,)]
          return action, arg
      if actionID == 1:  # [(128, 128)]
          arg1 = np.argmax(Target_Point[:128])
          arg2 = np.argmax(Target_Point[128:256])
          arg = [(arg1, arg2)]
          return action, arg
      if actionID == 2:  # [(2,), (128, 128)]
          arg1 = np.argmax(Queue)
          arg2 = np.argmax(Target_Point[256:384])
          arg3 = np.argmax(Target_Point[384:512])
          arg = [(arg1,), (arg2, arg3)]
          return action, arg
      if actionID == 3:  # [(2,), (128, 128)]
          arg1 = np.argmax(Queue)
          arg2 = np.argmax(Target_Point[:128])
          arg3 = np.argmax(Target_Point[128:256])
          arg = [(arg1,), (arg2, arg3)]
          return action, arg
      if actionID == 4:  # [(4,)]
          arg1 = np.argmax(Target_Unit[:4])
          arg = [(arg1,)]
          return action, arg
      if actionID == 5:  # [(2,)]
          arg1 = np.argmax(Queue)
          arg = [(arg1,)]
          return action, arg
      if actionID == 6:  # [(4,), (128, 128)]
          arg1 = np.argmax(Target_Unit[:4])
          arg2 = np.argmax(Target_Point[256:384])
          arg3 = np.argmax(Target_Point[384:512])
          arg = [(arg1,), (arg2, arg3)]
          return action, arg
      if actionID == 7:  # [(500,)]
          arg1 = np.argmax(Selected_Units)
          arg = [(arg1,)]
          return action, arg
      if actionID == 8:  # []
          arg = []
          return action, arg
      if actionID == 9:  # [(5,), (10,)]
          arg1 = np.argmax(Target_Unit[4:9])
          arg2 = np.argmax(Target_Unit[9:19])
          arg = [(arg1,), (arg2,)]
          return action, arg
      if actionID == 10:  # [(2,), (128, 128), (128, 128)]
          arg1 = np.argmax(Queue)
          arg2 = np.argmax(Target_Point[256:384])
          arg3 = np.argmax(Target_Point[384:512])
          arg4 = np.argmax(Target_Point[512:640])
          arg5 = np.argmax(Target_Point[640:768])
          arg = [(arg1,), (arg2, arg3), (arg4, arg5)]
          return action, arg
      if actionID == 11:  # [(10,)]
          arg1 = np.argmax(Target_Unit[9:19])
          arg = [(arg1,)]
          return action, arg

      return

  def update(self, observation, func, args):
    input_data = obs_pre_processing(observation)
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

    self.full_network.update(np.reshape(input_data, [1, self.full_network.input_size]), np.array(output_data))

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

  def train_network(self, input_data, available_actions):
      full_Qs = self.full_network.predict(input_data)
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