import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

i = keras.layers.Input(shape=(1,1,3))
o = keras.layers.Conv2D(filters=10, kernel_size=1)(i)

model = keras.Model(inputs=i, outputs=o)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy())

ia = np.ones(shape=(1, 1, 1, 3))
print(model.predict(ia))

traindatagen = ImageDataGenerator(rescale=1./255)
train_gen = traindatagen.flow_from_directory()
traindatagen.flow_from_dataframe()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(300, 625),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow test images in batches of 20 using test_datagen generator
test_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(300, 625),
        batch_size=20,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=1,
      validation_data=test_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)


model.save_weights('.\\checkpoints\\sc_checkpoint')