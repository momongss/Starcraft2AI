import tensorflow as tf

# Total params: 1,611,857
# Trainable params: 1,609,809
# Non-trainable params: 2,048

def target_point_encoder(autoregressive_embedding, action_type, map_skip):
    resblock_channels = 128
    resblock_kernel = 3
    resblock_layer = 4

    # autoregressive_embedding = layers.Input(shape=(1024,))
    # action_type = layers.Input(shape=(1,))
    # map_skip = layers.Input(shape=(16, 16, 128))

    embedding_skip = tf.compat.v1.keras.layers.Reshape(target_shape=(16, 16, 4))(autoregressive_embedding)
    map_skip_input = tf.compat.v1.keras.layers.concatenate([map_skip, embedding_skip])

    map_relu = tf.compat.v1.keras.activations.relu(map_skip_input)
    map_conv = tf.compat.v1.keras.layers.Conv2D(128, 1, activation='relu')(map_relu)

    for _ in range(resblock_layer):
        r = tf.compat.v1.keras.layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(map_conv)
        r = tf.compat.v1.keras.layers.BatchNormalization()(r)
        r = tf.compat.v1.keras.layers.Activation('relu')(r)
        r = tf.compat.v1.keras.layers.Conv2D(filters=resblock_channels, kernel_size=resblock_kernel, padding='same')(r)
        r = tf.compat.v1.keras.layers.BatchNormalization()(r)
        map_conv = tf.compat.v1.keras.layers.add([r, map_skip])

    minimap_conv = tf.compat.v1.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    minimap_conv = tf.compat.v1.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(minimap_conv)
    minimap_conv = tf.compat.v1.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same')(minimap_conv)
    minimap_logits = tf.compat.v1.keras.layers.Conv2DTranspose(1, 4, strides=1, padding='same')(minimap_conv)
    minimap_logits = tf.compat.v1.keras.layers.Reshape((128, 128), name='minimap_logits')(minimap_logits)

    map1_conv = tf.compat.v1.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    map1_conv = tf.compat.v1.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(map1_conv)
    map1_conv = tf.compat.v1.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same')(map1_conv)
    screen1_logits = tf.compat.v1.keras.layers.Conv2DTranspose(1, 4, strides=1, padding='same')(map1_conv)
    screen1_logits = tf.compat.v1.keras.layers.layers.Reshape((128, 128), name='screen1_logits')(screen1_logits)

    map2_conv = tf.compat.v1.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(map_conv)
    map2_conv = tf.compat.v1.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(map2_conv)
    map2_conv = tf.compat.v1.keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same')(map2_conv)
    screen2_logits = tf.compat.v1.keras.layers.Conv2DTranspose(1, 4, strides=1, padding='same')(map2_conv)
    screen2_logits = tf.compat.v1.keras.layers.Reshape((128, 128), name='screen2_logits')(screen2_logits)

    # model = models.Model(inputs=[autoregressive_embedding, action_type, map_skip], outputs=[minimap_logits, screen1_logits, screen2_logits], name='spatial_encoder')
    return minimap_logits, screen1_logits, screen2_logits


if __name__ == '__main__':
    pass