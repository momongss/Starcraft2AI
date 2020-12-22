import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

def model_test():
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_out = LSTM(32)(x)

    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    auxiliary_input = Input(shape=(5,), name='aux_input')
    x = tf.keras.layers.concatenate([lstm_out, auxiliary_input])
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(optimizer='adam', loss=tf.keras.losses.KLDivergence)
    model.summary()


model_test()