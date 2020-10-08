import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def multi_input_lstm_embedding_model(timesteps, columns_size, max_words, max_len):
    # lstm model
    lstm_input = layers.Input(shape=(timesteps, columns_size))
    lstm_out = layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3, name='lstm_out')(lstm_input)
    test_out = layers.Dense(100)(lstm_input)

    lstm_model = models.Model(inputs=lstm_input, outputs={'lstm_out': lstm_out, 'test_out': test_out})

    # embedding model
    embed_input = layers.Input(shape=(None,))
    embed_out = layers.Embedding(max_words, 8, input_length=max_len)(embed_input)
    embed_out = layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3))(embed_out)

    embed_model = models.Model(inputs=embed_input, outputs=embed_out)

    # concatenate
    print(lstm_model.output, type(lstm_model.output), len(lstm_model.output))
    concatenated = layers.concatenate([lstm_model.output['lstm_out'], embed_model.output])
    concatenated = layers.Dense(32, activation='relu')(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)
    concat_out = layers.Dense(2, activation='sigmoid')(concatenated)

    concat_model = models.Model([lstm_input, embed_input], concat_out)

    return concat_model


if __name__ == '__main__':
    model = multi_input_lstm_embedding_model(50, 10, 100, 100)
    model.summary()