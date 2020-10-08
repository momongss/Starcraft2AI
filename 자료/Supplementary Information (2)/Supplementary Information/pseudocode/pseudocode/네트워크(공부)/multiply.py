from tensorflow import keras
from tensorflow.keras import layers, models

layers.GRU

import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,1,1,1,1])
print(x.shape, y.shape)

# print(np.multiply(x, y))

a = layers.Input(shape=(5,))
b = layers.Input(shape=(5,))

c = layers.multiply([a,b])

model = models.Model(inputs=[a, b], outputs=[c])
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.KLDivergence())


model.summary()

print(model.predict(x=[[x], [y]]))