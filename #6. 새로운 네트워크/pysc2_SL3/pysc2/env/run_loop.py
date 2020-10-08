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
"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def run_loop(agents, env, max_frames=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup(obs_spec, act_spec)

  try:
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        # print(timesteps[0][3]['feature_minimap'].shape)
        # env = timesteps[0][3]
        # entity_list = env['feature_units']
        # minimap = env['feauture_minimap']
        #
        # # scalar features
        # single_select = env['single_select']
        # multi_select = env['multi_select']
        # build_queue = env['build_queue']
        # cargo = env['cargo']
        # production_queue = env['production_queue']
        # last_actions = env['last_actions']
        # cargo_slots_available = env['cargo_slots_available']
        # home_race_requested = env['home_race_requested']
        # away_race_requested = env['away_race_requested']
        # action_result = env['action_result']
        # alerts = env['alerts']
        # game_loop = env['game_loop']
        # score_cumulative = env['score_cumulative']
        # score_by_category = env['score_by_category']
        # score_by_vital = env['score_by_vital']
        # player = env['player']
        # control_groups = env['control_groups']
        # upgrades = env['upgrades']
        # available_actions_one_hot = env['available_actions_one_hot']

        # startcraft_model.predict(x=[])
        if max_frames and total_frames >= max_frames:
          return
        if timesteps[0].last():
          break
        timesteps = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
