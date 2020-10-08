import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

a = K.variable(1.0)
print(a.numpy())