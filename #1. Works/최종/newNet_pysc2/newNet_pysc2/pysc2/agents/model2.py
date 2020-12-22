# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random as ran

import os.path


# 꺼내서 사용할 리플레이 갯수
REPLAY = 1# 리플레이를 저장할 리스트
REPLAY_MEMORY = []
# 미니배치
MINIBATCH = 50

# 하이퍼파라미터
LEARNING_LATE = 0.1
NUM_EPISODE = 4000
e = 0.1
DISCOUNT = 0.9
rList = []


# 네트워크 클래스 구성
class NETWORK:
    def __init__(self, session, input_size, output_size, name="main"):
        # 네트워크 정보 입력
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        checkpoint = "./network_store/checkpoint"
        network_file_index = "./network_store/test_net.index"
        network_file_meta = "./network_store/test_net.meta"

        if os.path.isfile(checkpoint) and os.path.isfile(network_file_index) and os.path.isfile(network_file_meta):
            self.load_network()
            print("네트워크 로드 성공")
        else:
            self.build_network()
            print("네트워크 빌드 성공")

    def load_network(self, width=10, L_rate=1e-1):

        saver = tf.train.import_meta_graph('./network_store/test_net.meta')
        saver.restore(self.session, './network_store')

        graph = tf.get_default_graph()

        # 네트워크 구조
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.input_size])

        self.W1 = graph.get_tensor_by_name("w1:0")
        self.W2 = graph.get_tensor_by_name("w2:0")

        self.L1 = tf.nn.tanh(tf.matmul(self.x, self.W1))

        self.Q_pre = tf.matmul(self.L1, self.W2)
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.output_size))

        # 손실 함수
        self.loss = tf.reduce_sum(tf.square(self.y - self.Q_pre))
        self.train = tf.train.AdamOptimizer(learning_rate=L_rate).minimize(self.loss)

    def build_network(self, width=10, L_rate=1e-1):

        reshape_map = {"single_select": (1, 7),
                       "multi_select": (1, 1400),
                       "build_queue": (1, 140),
                       "cargo": (1, 140),
                       "production_queue": (1, 10),
                       "last_actions": (1, 1),
                       "cargo_slots_available": (1, 1),
                       "home_race_requested": (1, 1),
                       "away_race_requested": (1, 1),
                       "feature_screen": (1, 190512),
                       "feature_minimap": (1, 45056),
                       "action_result": (1, 1),
                       "alerts": (1, 2),
                       "game_loop": (1, 1),
                       "score_cumulative": (1, 13),
                       "score_by_category": (1, 55),
                       "score_by_vital": (1, 9),
                       "player": (1, 11),
                       "control_groups": (1, 20),
                       "upgrades": (1, 1),
                       "available_actions_one_hot": (1, 573)}

        W_map = {"single_select": (7 ,width),
                       "multi_select": (1400 ,width),
                       "build_queue": (140, width),
                       "cargo": (140, width),
                       "production_queue": (10, width),
                       "last_actions": (1, width),
                       "cargo_slots_available": (1, width),
                       "home_race_requested": (1, width),
                       "away_race_requested": (1, width),
                       "feature_screen": (190512, width),
                       "feature_minimap": (45056, width),
                       "action_result": (1, width),
                       "alerts": (2, width),
                       "game_loop": (1, width),
                       "score_cumulative": (13, width),
                       "score_by_category": (55, width),
                       "score_by_vital": (9, width),
                       "player": (11, width),
                       "control_groups": (20, width),
                       "upgrades": (1, width),
                       "available_actions_one_hot": (573, width)}

        # 네트워크 구조
        self.input = []
        for value in reshape_map.values():
            self.input.append(tf.placeholder(dtype=tf.float32, shape=value))

        self.W = []
        for key, value in W_map.items():
            self.W.append(tf.Variable(tf.random_uniform(shape=value)))

        self.B = []
        for value in W_map.values():
            self.B.append(tf.Variable(tf.random_uniform(shape=(1, width))))

        self.L = []
        for i in range(len(self.B)):
            self.L.append(tf.nn.tanh(tf.matmul(self.input[i], self.W[i]) + self.B[i]))

        self.output = self.L[0]
        for l in self.L[1:]:
            self.output += l

        self.output_w = tf.Variable(tf.random_uniform([width-1, self.output_size], 0, 0.01))

        self.Q_pre = tf.matmul(self.output, self.output_w)
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.output_size))

        # 손실 함수
        self.loss = tf.reduce_sum(tf.square(self.y - self.Q_pre))
        self.train = tf.train.AdamOptimizer(learning_rate=L_rate).minimize(self.loss)

    def test_update(self, input_data):
        # for i, value in enumerate(input_data.values()):
        #     self.session.run(self.L[i], feed_dict={self.input[i]: value})
        self.session.run(self.train,
                         feed_dict={self.input[0]: input_data['single_select'],
                                    self.input[1]: input_data['multi_select'],
                                    self.input[2]: input_data['build_queue'],
                                    self.input[3]: input_data['cargo'],
                                    self.input[4]: input_data['production_queue'],
                                    self.input[5]: input_data['last_actions'],
                                    self.input[6]: input_data['cargo_slots_available'],
                                    self.input[7]: input_data['home_race_requested'],
                                    self.input[8]: input_data['away_race_requested'],
                                    self.input[9]: input_data['feature_screen'],
                                    self.input[10]: input_data['feature_minimap'],
                                    self.input[11]: input_data['action_result'],
                                    self.input[12]: input_data['alerts'],
                                    self.input[13]: input_data['game_loop'],
                                    self.input[14]: input_data['score_cumulative'],
                                    self.input[15]: input_data['score_by_category'],
                                    self.input[16]: input_data['score_by_vital'],
                                    self.input[17]: input_data['player'],
                                    self.input[18]: input_data['control_groups'],
                                    self.input[19]: input_data['upgrades'],
                                    self.input[20]: input_data['available_actions_one_hot'],
                                    self.y: np.zeros(self.output_size).reshape(1, self.output_size)})
        print("test 성공")
        return

    # 예측한 Q값 구하기
    def predict(self, state):
        s_t = state.reshape(1, self.input_size)
        return self.session.run(self.Q_pre, feed_dict={self.x: s_t})

    # 네트워크 학습
    def update(self, x, y):
        self.session.run(self.train, feed_dict={self.x: x, self.y: y.reshape(1, 1558)})

    # 미니배치를 이용한 학습
    def replay_train(DQN, replay_memory, replay):
        for state, action, reward, next_state, done in ran.sample(replay_memory, replay):
            Q = DQN.predict(state)

            # DQN 알고리즘으로 학습
            if done:
                Q[0, action] = -100
            else:
                Q[0, action] = reward + DISCOUNT * np.max(DQN.predict(next_state))

            DQN.update(np.reshape(state, [1, DQN.input_size]), Q)
