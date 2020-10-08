from tensorflow import keras
from tensorflow.keras import layers, models

# action type head
# resblock 버전.
# convolution layer의 width를 얼마로 해야하는지 정보가 없음. ??

# Total params: 4,105,788

# input : lstm_output, scalar_context
# output : action_type_logits


def action_type():
    conv_width = 384 + 128
    action_types = 572

    lstm_output = keras.layers.Input(shape=(384, 1), name='lstm_output')
    scalar_context = keras.Input(shape=(128,), name='scalar_context')

    x = layers.Conv1D(conv_width, 3, padding='same')(lstm_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_1_output = layers.MaxPooling1D(3)(x)

    x = layers.Conv1D(conv_width, 3, padding='same')(block_1_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_2_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_3_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_4_output = layers.add([x, block_3_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_4_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_5_output = layers.add([x, block_4_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_5_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_6_output = layers.add([x, block_5_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_6_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_7_output = layers.add([x, block_6_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_7_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_8_output = layers.add([x, block_7_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_8_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_9_output = layers.add([x, block_8_output])

    x = layers.Conv1D(conv_width, 3, padding='same')(block_9_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(conv_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_10_output = layers.add([x, block_9_output])

    x = layers.Conv1D(conv_width, 3, activation='relu')(block_10_output)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(action_types, activation='softmax')(x)
    outputs = layers.Dropout(0.5)(x)

    model = models.Model(inputs=[lstm_output, scalar_context], outputs=[outputs], name='toy_resnet')
    return model


model = action_type()
model.summary()