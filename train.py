import matplotlib
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
import model
import data


if __name__ == "__main__":
    classifier = model.GarbageClassifier()
    dm = data.GarbageDataModule(batch_size=64)
    dm.setup()

    val_samples = next(iter(dm.train_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10,2))
    imgs_pl = val_imgs[:5].numpy().transpose((0, 2, 3, 1))

    for i, (img, label) in enumerate(zip(imgs_pl, val_labels)):
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(f"Label = {dm.train_dataloader().dataset.classes[label.item()]}")
    
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNjNhOTY3ZS1mOGU2LTQ2ZGItYTFmOS01MGY4ZDdiNGU1YTcifQ==",
        project="fedorgrab/garbage",
        name="Resnet Pytorch Lightning My Data",
    )
    neptune_logger.experiment["dataset_sample"].upload(fig)

    trainer = pl.Trainer(
        max_epochs=4,
        progress_bar_refresh_rate=20,
        gpus=1,
        logger=neptune_logger,
    )

    trainer.fit(classifier, dm)

    torch.save(classifier.state_dict(), "new_model.pt")
    neptune_logger.experiment["model"].upload("new_model.pt")
