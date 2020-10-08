import tensorflow as tf

A = tf.constant([2, 20, 30, 3, 6])
print(A.shape, tf.math.argmax(A))  # A[2] is maximum in tensor A

B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8],
                 [14, 45, 23, 5, 27]])

print(tf.math.argmax(B, 0))
print(tf.math.argmax(B, 1))