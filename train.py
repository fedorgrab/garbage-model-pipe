import torch
import pytorch_lightning as pl
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
import model
import data
import utils
import splitfolders
import constants


if __name__ == "__main__":
    classifier = model.GarbageClassifier(num_classes=constants.NUM_CLASSES)
    dm = data.GarbageDataModule(batch_size=constants.BATCH_SIZE)
    dm.setup()

    dataset_sample_fig = utils.get_dataset_sample(dm)

    neptune_logger = NeptuneLogger(
        project="fedorgrab/garbage",
        name="Resnet Pytorch Lightning My Data",
        api_key=constants.NEPTUNE_TOKEN,
    )
    neptune_logger.experiment["dataset_sample"].upload(dataset_sample_fig)

    trainer = pl.Trainer(
        max_epochs=constants.NUM_EPOCHS,
        progress_bar_refresh_rate=20,
        log_every_n_steps=7,
        gpus=constants.GPUS,
        logger=neptune_logger,
    )
    trainer.fit(classifier, dm)

    torch.save(classifier.state_dict(), "model.pt")
    neptune_logger.experiment["model"].upload("model.pt")
