import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 3,973,632

# LSTM
# input : embedded_entity, embedded_spatial, embedded_scalar
# output : LSTM output (256)


def core():
    embedded_scalar = layers.Input(shape=(608,), dtype='float32')
    embedded_entity = layers.Input(shape=(256,), dtype='float32')
    embedded_spatial = layers.Input(shape=(256,), dtype='float32')
    # 1120

    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    LSTM_input = layers.Reshape((1, 1120))(LSTM_input)
    # x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True)(x)
    lstm_out = layers.LSTM(384, return_state=False)(x)

    # x = tf.compat.v1.keras.layers.CuDNNLSTM(384, return_sequences=True)(x)
    # x = tf.compat.v1.keras.layers.CuDNNLSTM(384, return_sequences=True)(x)
    # lstm_out = tf.compat.v1.keras.layers.CuDNNLSTM(384, return_state=True)(x)
    print(lstm_out, 'lstm')
    print(len(lstm_out))

    LSTM_model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial], outputs=lstm_out)
    return LSTM_model


if __name__ == '__main__':
    import numpy as np
    embedded_scalar = np.random.rand(1, 608)
    embedded_entity = np.random.rand(1, 256)
    embedded_spatial = np.random.rand(1, 256)
    print(embedded_scalar.shape, 'e')
    print(embedded_entity.shape, 'e')
    print(embedded_spatial.shape, 'e')

    Core = core()
    Core.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.KLDivergence())

    predicted = Core.predict([embedded_scalar, embedded_entity, embedded_spatial])
    # for p in predicted:
    #     print(p.shape)
    print(predicted)
