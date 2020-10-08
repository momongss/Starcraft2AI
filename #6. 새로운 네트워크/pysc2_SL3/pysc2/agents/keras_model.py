# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

import numpy as np
import random as random
import os.path

# 하이퍼파라미터
LEARNING_LATE = 0.1
MINIBATCH = 50

# 네트워크 클래스 구성
class DQN:
    def __init__(self, input_size, output_size):
        # 네트워크 정보 입력
        self.input_size = input_size
        self.output_size = output_size

        self.build_network()

    def build_keras_net(self, width=10, L_rate=1e-1):

        # Spatial Encoder
        minimap = Input(shape=(11, 64, 64))
        Spatial_encoder = tf.keras.applications.ResNet50(input_tensor=minimap,
                                                         include_top=True,
                                                         weights=None,
                                                         pooling='max')

        x = Spatial_encoder.output
        x = tf.keras.layers.Dense(1024, name='fully', init='uniform')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        main_input = Input(shape=(100,), dtype='int32', name='main_input')
        x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
        lstm_out = LSTM(32)(x)

        auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = tf.keras.layers.concatenate([lstm_out, auxiliary_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
        model.compile(optimizer=tf.keras.optimizers.Adam, loss=tf.keras.losses.KLDivergence, loss_weights=[])
        model.fit()

    def build_network(self, width=2, L_rate=1e-1):

        # 네트워크 구조
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.input_size])

        self.W1 = tf.Variable(tf.random_uniform([self.input_size, width], 0, 0.01), name="w1")
        self.W2 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w2")
        self.W3 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w3")
        self.W4 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w4")
        self.W5 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w5")
        self.W6 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w6")
        self.W7 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w7")
        self.W8 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w8")
        self.W9 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w9")
        self.W10 = tf.Variable(tf.random_uniform([width, self.output_size], 0, 0.01), name="w10")

        self.B1 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b1")
        self.B2 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b2")
        self.B3 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b3")
        self.B4 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b4")
        self.B5 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b5")
        self.B6 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b6")
        self.B7 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b7")
        self.B8 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b8")
        self.B9 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b9")

        self.L1 = tf.nn.tanh(tf.matmul(self.x, self.W1)+self.B1)
        self.L2 = tf.nn.tanh(tf.matmul(self.L1, self.W2+self.B2))
        self.L3 = tf.nn.tanh(tf.matmul(self.L2, self.W3+self.B3))
        self.L4 = tf.nn.tanh(tf.matmul(self.L3, self.W4+self.B4))
        self.L5 = tf.nn.tanh(tf.matmul(self.L4, self.W5+self.B5))
        self.L6 = tf.nn.tanh(tf.matmul(self.L5, self.W6+self.B6))
        self.L7 = tf.nn.tanh(tf.matmul(self.L6, self.W7+self.B7))
        self.L8 = tf.nn.tanh(tf.matmul(self.L7, self.W8+self.B8))
        self.L9 = tf.nn.tanh(tf.matmul(self.L8, self.W9+self.B9))

        self.Q_pre = tf.matmul(self.L9, self.W10)
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
        self.session.run(self.train, feed_dict={self.x: x, self.y: y.reshape(1, 1558)})

    # 미니배치를 이용한 학습
    def replay_train(DQN, replay_memory, replay):
        for state, action, reward, next_state, done in random.sample(replay_memory, replay):
            Q = DQN.predict(state)

            # DQN 알고리즘으로 학습
            if done:
                Q[0, action] = -100
            else:
                Q[0, action] = reward + DISCOUNT * np.max(DQN.predict(next_state))

            DQN.update(np.reshape(state, [1, DQN.input_size]), Q)