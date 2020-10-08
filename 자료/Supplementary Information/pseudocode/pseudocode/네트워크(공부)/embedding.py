from tensorflow import keras
from tensorflow.keras import layers, models

input = layers.Input(shape=(128, 8), dtype='float32')
output = layers.Embedding(input_dim=1, output_dim=1, input_length=384)(input)
model = models.Model(inputs=input, outputs=output)
model.summary()