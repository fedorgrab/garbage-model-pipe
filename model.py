import tensorflow
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import constants
from data_augmentation import data_augmentation


def create_model(train_dataset) -> tensorflow.keras.Model:
    base_model = MobileNet(
        input_shape=constants.IMAGE_SHAPE,
        include_top=False,
        weights="imagenet"
    )

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    base_model.trainable = False

    global_average_layer = keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    prediction_layer = keras.layers.Dense(9)
    prediction_batch = prediction_layer(feature_batch_average)

    inputs = keras.Input(shape=constants.IMAGE_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
