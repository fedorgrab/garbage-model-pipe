import torch
import pytorch_lightning as pl
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
import model
import data
import utils
import splitfolders

if __name__ == "__main__":
    classifier = model.GarbageClassifier(num_classes=9)
    dm = data.GarbageDataModule(batch_size=64)
    dm.setup()

    dataset_sample_fig = utils.get_dataset_sample(dm)

    neptune_logger = NeptuneLogger(
        project="fedorgrab/garbage",
        name="Resnet Pytorch Lightning My Data",
    )
    neptune_logger.experiment["dataset_sample"].upload(dataset_sample_fig)

    trainer = pl.Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=20,
        gpus=0,
        logger=neptune_logger,
    )

    trainer.fit(classifier, dm)

    torch.save(classifier.state_dict(), "model.pt")
    neptune_logger.experiment["model"].upload("model.pt")
