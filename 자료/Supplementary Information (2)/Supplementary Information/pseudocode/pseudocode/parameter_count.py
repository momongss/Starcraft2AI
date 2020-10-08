import tensorflow as tf

input_size = 237000
width = 1000
output_size = 1500

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
W10 = tf.Variable(tf.random_uniform([width, output_size], 0, 0.01), name="w10")

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)