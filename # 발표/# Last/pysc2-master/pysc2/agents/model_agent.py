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
"""A model agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.agents.Network.architecture import get_action_Model, get_point_Model


class ModelAgent(base_agent.BaseAgent):
  """A model agent for starcraft."""
  def __init__(self):
      super(ModelAgent, self).__init__()
      self.action_model = get_action_Model()
      self.point_model = get_point_Model()

      self.DISCOUNT = 0.9
      self.action_id = 0
      self.point = 0
      self.valid_actions = {}

  def state2input(self, obs):
      obs.observation['feature_minimap'] = np.array(list(obs.observation['feature_minimap'])).reshape(1, 128, 128, 11)
      obs.observation['player'] = np.array(list(obs.observation['player'])).reshape(1, 11)
      obs.observation['game_loop'] = np.array(list(obs.observation['game_loop'])).reshape(1, 1)
      return {'feature_minimap': obs.observation['feature_minimap'],
              'player': obs.observation['player'],
              'game_loop': obs.observation['game_loop']}

  def train_action_Q(self, next_state, reward, done, state):
      Qf = self.action_model.predict(x=self.state2input(state))
      if done:
          Qf[0][self.action_id] = reward
      else:
          next_Qf = self.action_model.predict(x=self.state2input(next_state))

          print(next_Qf)
          print(Qf)
          Qf[0][self.action_id] = reward + self.DISCOUNT * np.max(next_Qf[0])
      print('action id:', self.action_id, 'Q:', Qf)
      self.action_model.fit(x=self.state2input(state), y=Qf)
      return

  def train_point_Q(self, next_state, reward, done, state):
      Qp = self.action_model.predict(x=self.state2input(state))
      if done:
          Qp[0][self.point] = reward
      else:
          next_Qf = self.action_model.predict(x=self.state2input(next_state))

          Qp[0][self.point] = reward + self.DISCOUNT * np.max(next_Qf[0])
      print('point id:', self.point, 'Q:', Qp)
      self.action_model.fit(x=self.state2input(state), y=Qp)
      return

  def step(self, obs):
    super(ModelAgent, self).step(obs)
    function_id = np.random.choice(obs.observation.available_actions)
    args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    print(obs.reward)
    return actions.FunctionCall(function_id, args)