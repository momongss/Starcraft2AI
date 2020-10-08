import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from architecture import get_action_Model, get_point_Model


action_model = get_action_Model()
point_model = get_point_Model()

