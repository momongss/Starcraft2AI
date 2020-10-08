import tensorflow as tf

input_size = 270000
output_size = 1557
width = 1000

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, input_size])

W1 = tf.Variable(tf.random_uniform([input_size, width], 0, 0.01), name="w1")
W2 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w2")
W3 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w3")
W4 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w4")
W5 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w5")
W6 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w6")
W7 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w7")
W8 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w8")
W9 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w9")
W10 = tf.Variable(tf.random_uniform([width, width], 0, 0.01), name="w10")


B1 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b1")
B2 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b2")
B3 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b3")
B4 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b4")
B5 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b5")
B6 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b6")
B7 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b7")
B8 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b8")
B9 = tf.Variable(tf.random_uniform([width], 0, 0.01), name="b9")

L1 = tf.nn.tanh(tf.matmul(x, W1)+B1)
L2 = tf.nn.tanh(tf.matmul(L1, W2)+B2)
L3 = tf.nn.tanh(tf.matmul(L2, W3)+B3)
L4 = tf.nn.tanh(tf.matmul(L3, W4)+B4)
L5 = tf.nn.tanh(tf.matmul(L4, W5)+B5)
L6 = tf.nn.tanh(tf.matmul(L5, W6)+B6)
L7 = tf.nn.tanh(tf.matmul(L6, W7)+B7)
L8 = tf.nn.tanh(tf.matmul(L7, W8)+B8)
L9 = tf.nn.tanh(tf.matmul(L8, W9)+B9)

Q_pre = tf.matmul(L9, W10)

y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, output_size))

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(variable.name)
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)

        # 손실 함수
        # self.loss = tf.reduce_sum(tf.square(y - self.Q_pre))
        # self.train = tf.train.AdamOptimizer(learning_rate=L_rate).minimize(self.loss)