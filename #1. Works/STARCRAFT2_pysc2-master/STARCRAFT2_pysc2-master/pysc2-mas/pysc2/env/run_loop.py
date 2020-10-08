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
from collections import deque
import numpy as np
import random
import tensorflow as tf

import pickle

MINIBATCH = 100
REPLAY = 1
DISCOUNT = 0.9


def state_processing(obs):
  input_data = []
  for key, value in obs.observation.items():
    if key != 'map_name':
      tmp = list(obs.observation[key].flatten())
      while len(tmp) < 100:
        tmp.append(0)
      input_data.extend(tmp)
  return input_data


# 미니배치를 이용한 학습
def replay_train(agent, replay_memory, replay):
  DQN = agent.full_network
  for next_state, action, reward, done, state in random.sample(replay_memory, replay):
    next_state = np.array(next_state).reshape(1, len(next_state))
    state = np.array(state).reshape(1, len(state))
    Q = DQN.predict(state)

    # DQN 알고리즘으로 학습
    action_index = 573

    now = DQN.predict(state)
    args = [now[0][573:]]
    data_2 = np.argmax(args[0][0:2]) + action_index
    data_4 = np.argmax(args[0][2:6]) + action_index + 2
    data_5 = np.argmax(args[0][6:11]) + action_index + 6
    data_10 = np.argmax(args[0][11:21]) + action_index + 11
    data_500 = np.argmax(args[0][21:521]) + action_index + 21
    data_84_1_0 = np.argmax(args[0][521:605]) + action_index + 521
    data_84_1_1 = np.argmax(args[0][605:689]) + action_index + 605
    data_84_2_0 = np.argmax(args[0][689:773]) + action_index + 689
    data_84_2_1 = np.argmax(args[0][773:857]) + action_index + 773
    data_64_0 = np.argmax(args[0][857:921]) + action_index + 857
    data_64_1 = np.argmax(args[0][921:985]) + action_index + 921

    next = DQN.predict(next_state)

    train2 = False
    train4 = False
    train5 = False
    train10 = False
    train500 = False
    train6464 = False
    train8484 = 0

    for arg in agent.action_spec.functions[action].args:
      if arg.sizes == (2,):
        train2 = True
      if arg.sizes == (4,):
        train4 = True
      if arg.sizes == (5,):
        train5 = True
      if arg.sizes == (10,):
        train10 = True
      if arg.sizes == (500,):
        train500 = True
      if arg.sizes == (64, 64):
        train6464 = True
      if arg.sizes == (84, 84):
        train8484 += 1

    next_args = [next[0][573:]]

    Q[0, action] = reward + DISCOUNT * np.max(next[0][0:573])
    if train2 : Q[0, data_2] = reward + DISCOUNT * np.max(next_args[0][0:2])
    if train4 : Q[0, data_4] = reward + DISCOUNT * np.max(next_args[0][2:6])
    if train5 : Q[0, data_5] = reward + DISCOUNT * np.max(next_args[0][6:11])
    if train10 : Q[0, data_10] = reward + DISCOUNT * np.max(next_args[0][11:21])
    if train500 : Q[0, data_500] = reward + DISCOUNT * np.max(next_args[0][21:521])
    if train8484 >= 1 : Q[0, data_84_1_0] = reward + DISCOUNT * np.max(next_args[0][521:605])
    if train8484 >= 1 : Q[0, data_84_1_1] = reward + DISCOUNT * np.max(next_args[0][605:689])
    if train8484 == 2 : Q[0, data_84_2_0] = reward + DISCOUNT * np.max(next_args[0][689:773])
    if train8484 == 2 : Q[0, data_84_2_1] = reward + DISCOUNT * np.max(next_args[0][773:857])
    if train6464 : Q[0, data_64_0] = reward + DISCOUNT * np.max(next_args[0][857:921])
    if train6464 : Q[0, data_64_1] = reward + DISCOUNT * np.max(next_args[0][921:985])

    print("액션 :", action, "Q값 :", Q[0, action])

    DQN.update(np.reshape(state, [1, DQN.input_size]), Q)


def replay_train2(agent, replay_memory, replay):
  DQN = agent.full_network
  for next_state, action, reward, done, state in random.sample(replay_memory, replay):
    next_state = np.array(next_state).reshape(1, len(next_state))
    state = np.array(state).reshape(1, len(state))
    Q = DQN.predict(state)

    next_Q = DQN.predict(next_state)

    Q[0, action] = reward + DISCOUNT * np.max(next_Q[0][:573])
    print("액션 :", action, "Q값 :", Q[0, action])

    func_id = np.argmax(next_Q[0][:573])
    args_Qs = [DQN.predict(next_state)[0][573:]]
    for arg in agent.action_spec.functions[func_id].args:
      if arg.sizes:
        pass

    argument = agent.arguments_processing(func_id, args_Qs)

    DQN.update(np.reshape(state, [1, DQN.input_size]), Q)


def run_loop(agents, env, max_frames=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""

  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  # print()
  # print("observation_space :", observation_spec)
  # print()
  # print("action_space :", action_spec)
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    # print("act spaces :",act_spec, type(act_spec))
    agent.setup(obs_spec, act_spec)

  # 핵심 변수
  # env : pysc2가 제공하는 스타크래프트 게임 환경 객체
  # agents : 강화학습을 위한 에이전트
  # timesteps : env.step() 메소드의 리턴값.
  #             env 을 agents 전달하기 좋은 형태로 만든 것으로 보인다.

  # 0. env 리셋
  #    agents 리셋
  # loop
  #   1. timesteps(processed env) -> agents에 전달 => action 결정
  #   2. action -> env에 반영 => timesteps 생성. :: 환경 변화

  # 결론 : timesteps에서 적합한 action 을 찾아내는게 현재 해야할 일.(현재 이게 완전히 랜덤)

  saver = tf.train.Saver()

  try:
    replay_buffer = deque(maxlen=100)
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      state = env.reset()   # 0. 초기 환경
      for a in agents:          # 0. agent 초기화
        a.reset()
      while True:

        actions = [agent.step(timestep, epsilon=0.1)
                   for agent, timestep in zip(agents, state)]
        function_id = actions[0][1]
        args = actions[0][2]
        actions = [actions[0][0]]

        next_state = env.step(actions)
        reward = next_state[0][1]
        done = next_state[0].last()

        # print("frame: ", total_frames , "function_id: ", function_id, " args: ", args)

        replay_buffer.append((state_processing(next_state[0]), function_id, reward, done, state_processing(state[0])))

        if total_frames % 10 == 99:
          for _ in range(MINIBATCH):
            replay_train(agents[0], replay_buffer, REPLAY)

        if total_frames % 100 == 99:
          saver.save(agents[0].sess, 'weight/DQN-model', global_step=total_frames)
          print(agents[0].sess.run(agents[0].full_network.W3))
          print("weight 저장됨")

        state = next_state
        total_frames += 1

        if max_frames and total_frames >= max_frames:
          return
        if next_state[0].last():
          break


  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))

# python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards