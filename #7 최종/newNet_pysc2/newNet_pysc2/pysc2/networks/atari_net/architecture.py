import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

entity_list = layers.Input(shape=(50, 46), name='entity_list')
minimap = layers.Input(shape=(128, 128, 11), name='minimap')
player = layers.Input(shape=(11,), name='player')

