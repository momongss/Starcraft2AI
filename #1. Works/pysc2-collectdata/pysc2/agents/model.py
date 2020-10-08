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
class DQN:
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

        # 네트워크 구조
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.input_size])

        self.W1 = tf.Variable(tf.random_uniform([self.input_size, width], 0, 0.01), name="w1")
        self.W2 = tf.Variable(tf.random_uniform([width, self.output_size], 0, 0.01), name="w2")

        self.L1=tf.nn.tanh(tf.matmul(self.x, self.W1))

        self.Q_pre = tf.matmul(self.L1, self.W2)
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.output_size))

        # 손실 함수
        self.loss = tf.reduce_sum(tf.square(self.y - self.Q_pre))
        self.train = tf.train.AdamOptimizer(learning_rate=L_rate).minimize(self.loss)

    # 예측한 Q값 구하기
    def predict(self, state):
        s_t = state.reshape(1, self.input_size)
        return self.session.run(self.Q_pre, feed_dict={self.x: s_t})

    # 네트워크 학습
    def update(self, x, y):
        self.session.run(self.train, feed_dict={self.x: x, self.y: y})


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


# # 메인
# def main():
#     with tf.Session() as sess:
#         # mainDQN 이라는 DQN 클래스 생성
#         mainDQN = DQN(sess, INPUT, OUTPUT)
#
#         # 변수 초기화
#         sess.run(tf.global_variables_initializer())
#         for step in range(NUM_EPISODE):
#             s = env.reset()
#             e = 1. / ((step/10)+1)
#             rall = 0
#             d = False
#             count=0
#
#             while not d and count < 5000:
#                 env.render()
#                 count+=1
#                 # e-greedy 를 사용하여 action값 구함
#                 if e > np.random.rand(1):
#                     a = env.action_space.sample()
#                 else: a = np.argmax(mainDQN.predict(s))
#
#                 # action을 취함
#                 s1, r, d, _ = env.step(a)
#
#                 # state, action, reward, next_state, done 을 메모리에 저장
#                 REPLAY_MEMORY.append([s,a,r,s1,d])
#
#                 # 메모리에 50000개 이상의 값이 들어가면 가장 먼저 들어간 것부터 삭제
#                 if len(REPLAY_MEMORY) > 50000:
#                     del REPLAY_MEMORY[0]
#
#                 rall += r
#                 s = s1
#
#             # 10 번의 스탭마다 미니배치로 학습
#             if step % 10 == 1 :
#                 for _ in range(MINIBATCH):
#                     replay_train(mainDQN,REPLAY_MEMORY,REPLAY)
#
#             rList.append(rall)
#             print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(step, count, rall, np.mean(rList)))
#
# if __name__ == '__main__':
#     main()
