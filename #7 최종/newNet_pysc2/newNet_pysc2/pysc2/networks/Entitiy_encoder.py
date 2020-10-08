import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 6,041,600
# Trainable params: 6,041,600
# Non-trainable params: 0

# input : entity_list[feature_units] (512, 46)
# output : entity_embeddings (256)


def entity_encoder(entity_list):
    # entity_list = layers.Input(shape=(500, 46), dtype='float32')
    entity_flatten = layers.Flatten()(entity_list)

    embedded_entity = layers.ReLU()(entity_list)
    embedded_entity = layers.Conv1D(filters=256, kernel_size=1)(embedded_entity)
    embedded_entity = layers.Flatten()(embedded_entity)

    embedded_entity = layers.Dense(256)(embedded_entity)
    embedded_entity = layers.Dense(256)(embedded_entity)

    entity_embeddings = layers.Dense(256, activation='relu')(entity_flatten)
    # model = models.Model(inputs=entity_list, outputs=[embedded_entity, entity_embeddings], name='entity_model')
    entity_embeddings = layers.Flatten()(entity_embeddings)
    return embedded_entity, entity_embeddings


if __name__ == '__main__':
    entity_model = entity_encoder()
    entity_model.summary()