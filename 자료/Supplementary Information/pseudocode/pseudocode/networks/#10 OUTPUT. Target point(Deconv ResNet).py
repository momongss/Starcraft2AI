from tensorflow import keras
from tensorflow.keras import layers, models


# Total params: 1,611,857
# Trainable params: 1,609,809
# Non-trainable params: 2,048

def spatial_encoder():
    resblock_channels = 128
    resblock_kernel = 3
    resblock_layer = 4

    autoregressive_embedding = layers.Input(shape=(1024,))
    action_type = layers.Input(shape=(1,))
    map_skip = layers.Input(shape=(16, 16, 128))

    embedding_skip = layers.Reshape(target_shape=(16, 16, 4))(autoregressive_embedding)
    map_skip_input = layers.concatenate([map_skip, embedding_skip])

    map_relu = keras.activations.relu(map_skip_input)
    map_conv = layers.Conv2D(128, 1, activation='relu')(map_relu)

    for _ in range(resblock_layer):
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(map_conv)
        r = layers.BatchNormalization()(r)
        r = layers.Activation('relu')(r)
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(r)
        r = layers.BatchNormalization()(r)
        map_conv = layers.add([r, map_skip])

    map_conv = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(map_conv)
    map_conv = layers.Conv2DTranspose(16, 4, strides=2, padding='same')(map_conv)
    target_location_logits = layers.Conv2DTranspose(1, 4, strides=2, padding='same')(map_conv)
    target_location_logits = keras.layers.Reshape((256, 256))(target_location_logits)

    model = models.Model(inputs=[autoregressive_embedding, action_type, map_skip], outputs=target_location_logits, name='spatial_encoder')
    return model


if __name__ == '__main__':
    spatial_model = spatial_encoder()
    spatial_model.summary()