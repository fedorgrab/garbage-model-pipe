from tensorflow import keras
from tensorflow.data import AUTOTUNE
import constants


def get_datasets():
    train_dataset = keras.utils.image_dataset_from_directory(
        constants.DATA_DIR,
        seed=123,
        validation_split=0.1,
        subset="training",
        shuffle=True,
        batch_size=constants.BATCH_SIZE,
        image_size=constants.IMAGE_SIZE,
    )
    validation_dataset = keras.utils.image_dataset_from_directory(
        constants.DATA_DIR,
        seed=123,
        validation_split=0.1,
        subset="validation",
        shuffle=False,
        batch_size=constants.BATCH_SIZE,
        image_size=constants.IMAGE_SIZE,
    )
    class_names = train_dataset.class_names
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, class_names
