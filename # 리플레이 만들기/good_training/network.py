import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Networks:
    def __init__(self):
        self.action_type_model = self.get_model()

    def get_model(self):
        entity_list = layers.Input(shape=(50, 46), name='entity_list', batch_size=1)
        minimap = layers.Input(shape=(128, 128, 11), name='minimap', batch_size=1)
        player = layers.Input(shape=(11,), name='player', batch_size=1)
        game_loop = layers.Input(shape=(1,), name='game_loop', batch_size=1)

