from tensorflow.python.client import device_lib
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import data
import model
import constants
import utils

if __name__ == "__main__":
    # Prepare data and environment
    # utils.unzip_dataset_archive()
    # utils.delete_bad_images()
    # Initialize Datasets
    print("DEVICES:")
    print(device_lib.list_local_devices())
    train_dataset, validation_dataset, class_names = data.get_datasets()
    # Initialize Model
    classifier = model.create_model(train_dataset=train_dataset)
    # Define Logger
    run = neptune.init(project="fedorgrab/garbage")
    neptune_cbk = NeptuneCallback(run=run, base_namespace="metrics")
    # Log Dataset Sample
    dataset_sample_fig = utils.get_dataset_sample(train_dataset, class_names)
    run["dataset_sample"].upload(dataset_sample_fig)
    # Train Classifier
    classifier.fit(
        train_dataset,
        epochs=constants.NUM_EPOCHS,
        validation_data=validation_dataset,
        callbacks=[neptune_cbk],
    )
    classifier.save("model.h5")
    run["model"].upload("model.h5")
