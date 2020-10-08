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
"""A base agent to write custom scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions

import tensorflow as tf
from pysc2.agents.model import DQN

class BaseAgent(object):
  """A base agent to write custom scripted agents.

  It can also act as a passive agent that does nothing but no-ops.
  """

  def make_network(self):
    input_size = 237468
    output_size = 573
    learning_rate = 0.1

    self.X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
    self.W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))


    self.Qpred = tf.matmul(self.X, self.W)
    self.Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

    self.loss = tf.reduce_sum(tf.square(self.Y - self.Qpred))

    self.train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

    self.dis = .99
    self.num_episodes = 2000

    self.rList = []

  def make_args_network(self):
    input_size = 237468 + 1
    output_size = 985
    learning_rate = 0.1

    self.args_X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
    self.args_W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

    self.args_Qpred = tf.matmul(self.args_X, self.args_W)
    self.args_Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

    self.args_loss = tf.reduce_sum(tf.square(self.args_Y - self.args_Qpred))

    self.args_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.args_loss)

    self.args_dis = .99
    self.args_num_episodes = 2000

    self.args_rList = []

  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = None
    self.action_spec = None

    # self.make_network()
    # self.make_args_network()

    init_op = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init_op)

    self.function_network = DQN(session=self.sess, input_size=237468, output_size=573, name="function_network")
    self.args_network = DQN(session=self.sess, input_size=237468 + 1, output_size=985, name="args_network")



  def setup(self, obs_spec, action_spec):
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def reset(self):
    self.episodes += 1

  def step(self, obs):
    self.steps += 1
    self.reward += obs.reward
    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])