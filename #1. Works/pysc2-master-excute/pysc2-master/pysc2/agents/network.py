import tensorflow as tf
import numpy as np
import random
import math


class simple_network(object):
    def __init__(self, input_size, output_size, hiddenSize):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hiddenSize
        # Parameters
        epsilon = 1  # The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
        epsilonMinimumValue = 0.001  # The minimum value we want epsilon to reach in training. (0 to 1)
        # output_size = 3  # The number of actions. Since we only have left/stay/right that means 3 actions.
        epoch = 1001  # The number of games we want the system to run for.
        # hiddenSize = 100  # Number of neurons in the hidden layers.
        maxMemory = 500  # How large should the memory be (where it stores its past experiences).
        batchSize = 50  # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
        gridSize = 10  # The size of the grid that the agent is going to play the game on.
        # input_size = gridSize * gridSize  # We eventually flatten to a 1d tensor to feed the network.
        discount = 0.9  # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)
        learningRate = 0.2  # Learning Rate for Stochastic Gradient Descent (our optimizer).

        # Create the base model.
        self.X = tf.placeholder(tf.float32, [None, input_size])
        self.W1 = tf.Variable(tf.truncated_normal([input_size, hiddenSize], stddev=1.0 / math.sqrt(float(input_size))))
        self.b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
        self.input_layer = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
        self.W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize], stddev=1.0 / math.sqrt(float(hiddenSize))))
        self.b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
        self.hidden_layer = tf.nn.relu(tf.matmul(self.input_layer, self.W2) + self.b2)
        self.W3 = tf.Variable(tf.truncated_normal([hiddenSize, output_size], stddev=1.0 / math.sqrt(float(hiddenSize))))
        self.b3 = tf.Variable(tf.truncated_normal([output_size], stddev=0.01))
        self.output_layer = tf.matmul(self.hidden_layer, self.W3) + self.b3

        # True labels
        self.Y = tf.placeholder(tf.float32, [None, output_size])

        # Mean squared error cost function
        self.loss = tf.reduce_sum(tf.square(self.Y - self.output_layer)) / (2 * batchSize)

        # Stochastic Gradient Decent Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)