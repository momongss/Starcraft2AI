import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Total params: 3,973,632

# LSTM
# input : embedded_entity, embedded_spatial, embedded_scalar
# output : LSTM output (256)


def core(embedded_scalar, embedded_entity, embedded_spatial):
    # embedded_scalar = layers.Input(shape=(608,), dtype='float32')
    # embedded_entity = layers.Input(shape=(256,), dtype='float32')
    # embedded_spatial = layers.Input(shape=(256,), dtype='float32')
    # total : 1120
    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    LSTM_input = layers.Reshape((1, 1920))(LSTM_input)

    # x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True, stateful=True)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True, stateful=True)(x)
    lstm_out = layers.LSTM(384, stateful=True)(x)

    return lstm_out

def core_lstm_cell(embedded_scalar, embedded_entity, embedded_spatial, h_state, c_state):
    # embedded_scalar = layers.Input(shape=(608,), dtype='float32')
    # embedded_entity = layers.Input(shape=(256,), dtype='float32')
    # embedded_spatial = layers.Input(shape=(256,), dtype='float32')
    # h_state = layers.Input(shape=(384,), dtype='float32')
    # c_state = layers.Input(shape=(384,), dtype='float32')
    # total : 1120
    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    # LSTM_input = layers.Reshape((1, 1120))(LSTM_input)
    lstm1 = layers.LSTMCell(384)
    lstm2 = layers.LSTMCell(384)
    lstm3 = layers.LSTMCell(384)
    lstm1.build(input_shape=(1, 2899))
    lstm2.build(input_shape=(1, 384))
    lstm3.build(input_shape=(1, 384))
    o, [h, c] = lstm1.call(LSTM_input, [h_state, c_state])
    o, [h, c] = lstm2.call(o, [h, c])
    o, [h, c] = lstm3.call(o, [h, c])

    h = layers.Reshape((384,), name="next_state_h")(h)
    c = layers.Reshape((384,), name="next_state_c")(c)

    # model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial, h_state, c_state], outputs=[o, h, c])
    # model.summary()

    return o, h, c


# if __name__ == '__main__':
#     embedded_scalar = layers.Input(shape=(608,), dtype='float32')
#     embedded_entity = layers.Input(shape=(256,), dtype='float32')
#     embedded_spatial = layers.Input(shape=(256,), dtype='float32')
#     h_state = layers.Input(shape=(384,), dtype='float32')
#     c_state = layers.Input(shape=(384,), dtype='float32')
#     # total : 1120
#     LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
#     # LSTM_input = layers.Reshape((1, 1120))(LSTM_input)
#     lstm1 = layers.LSTMCell(384)
#     lstm2 = layers.LSTMCell(384)
#     lstm3 = layers.LSTMCell(384)
#     lstm1.build(input_shape=(1, 1120))
#     lstm2.build(input_shape=(1, 384))
#     lstm3.build(input_shape=(1, 384))
#     o, [h, c] = lstm1.call(LSTM_input, [h_state, c_state])
#     o, [h, c] = lstm2.call(o, [h, c])
#     o, [h, c] = lstm3.call(o, [h, c])
#
#     h = layers.Reshape((384,), name="next_state_h")(h)
#     c = layers.Reshape((384,), name="next_state_c")(c)
#
#     model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial, h_state, c_state], outputs=[o, h, c])
#     model.summary()

if __name__ == '__main__':
    embedded_scalar = layers.Input(shape=(608,), dtype='float32', batch_size=30)
    embedded_entity = layers.Input(shape=(256,), dtype='float32', batch_size=30)
    embedded_spatial = layers.Input(shape=(256,), dtype='float32', batch_size=30)
    # total : 1120
    LSTM_input = layers.concatenate([embedded_scalar, embedded_entity, embedded_spatial])
    LSTM_input = layers.Reshape((1, 1120))(LSTM_input)

    # x = layers.Embedding(input_dim=1120, output_dim=384)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True, stateful=True)(LSTM_input)
    x = layers.LSTM(384, return_sequences=True, stateful=True)(x)
    lstm_out = layers.LSTM(384, stateful=True)(x)

    model = models.Model(inputs=[embedded_scalar, embedded_entity, embedded_spatial],
                         outputs=[lstm_out])
    model.summary()

def core_sequential():
    input_size = 384
    hidden_size = 384
    model = models.Sequential()

    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))
    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))
    model.add(layers.LSTM(hidden_size, input_shape=(input_size, 1), return_sequences=True))

    model.compile(optimizer='adam', loss=keras.losses.KLDivergence())
    return model




# import numpy as np
#
# a = np.array([[[1], [2], [3]]])
# print(a.shape)
# print(model.predict(a))