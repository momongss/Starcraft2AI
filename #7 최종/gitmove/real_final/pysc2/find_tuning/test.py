from tensorflow import keras

mobileNet = keras.applications.mobilenet_v2(
    weights='imagenet',
    input_shape=(128,128,11),
    include_top=False)

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.