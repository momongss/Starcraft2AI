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


def select_function(available_actions):
  return np.random.choice(available_actions)


def select_args(size):
  return np.random.randint(0, size)


def one_hot(x):
  return np.identity(235685)[x:x + 1]


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


    # # Qs2 = self.function_network.update(func_input_data, Qs)
    # functions = Qs[0]
    # for i in range(len(functions)):
    #   if i not in available_actions:
    #     functions[i] = 0
    #
    # func_id = np.argmax(Qs)
    # input_data.append(func_id)
    # args_input_data = np.array(input_data).reshape(1, len(input_data))
    #
    # args_Qs = self.args_network.predict(args_input_data)
    # arguments = self.arguments_processing(func_id, args_Qs)
    #
    # return func_id, arguments

  def step(self, obs, epsilon):
    super(ModelAgent, self).step(obs)
    input_data = []
    for key, value in obs.observation.items():
      if key != 'map_name':
        tmp = list(obs.observation[key].flatten())
        while len(tmp) < 100:
          tmp.append(0)
        input_data.extend(tmp)
    print(len(input_data))

    if np.random.random() < epsilon:
        function_id = select_function(obs.observation.available_actions)
        args = [[select_args(size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
    else:
        function_id, args = self.train_network(input_data, set(obs.observation.available_actions))

    return actions.FunctionCall(function_id, args), function_id, args