import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


def get_compiled_Model():
    # Input
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = get_compiled_Model()
    model.summary()