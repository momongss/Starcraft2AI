from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 10,000,320
# Trainable params: 9,998,272
# Non-trainable params: 2,048

# input : minimap[feature_minimap] (128, 128, 11)
# output : map_skip (16, 16, 128), embedded_spatial (256)


def spatial_encoder(minimap):
    resblock_channels = 128
    resblock_kernel = 3
    resblock_layer = 4
    
    # minimap = layers.Input(shape=(128, 128, 11))
    c = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(minimap)
    c = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(c)
    c = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(c)
    map_skip = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(c)

    for _ in range(resblock_layer):
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(map_skip)
        r = layers.BatchNormalization()(r)
        r = layers.Activation('relu')(r)
        r = layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(r)
        r = layers.BatchNormalization()(r)
        map_skip = layers.add([r, map_skip])

    map_flatten = layers.Flatten()(map_skip)
    embedded_spatial = layers.Dense(256, activation='relu', name='embedded_spatial')(map_flatten)
    embedded_spatial = layers.Flatten()(embedded_spatial)

    # model = models.Model(inputs=minimap, outputs=[map_skip, embedded_spatial], name='spatial_model')
    return map_skip, embedded_spatial


if __name__ == '__main__':
    spatial_model = spatial_encoder()
    spatial_model.summary()