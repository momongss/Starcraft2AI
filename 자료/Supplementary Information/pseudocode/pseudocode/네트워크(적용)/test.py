from tensorflow import keras
from tensorflow.keras import layers, models


def spatial_encoder():
    minimap = layers.Input(shape=(64, 64, 11))
    output = layers.Conv2D(filters=3, kernel_size=1, padding='valid', activation='relu')(minimap)
    keras.applications.NASNetLarge(in)
    # conv16 = layers.MaxPool2D(pool_size=(4, 4), strides=2, padding='valid')(output)
    # resnet = keras.applications.ResNet50(include_top=False, input_tensor=output, classes=100)
    # o = resnet(output)
    model = models.Model(inputs=minimap, outputs=o)
    model.summary()


spatial_encoder()