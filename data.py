from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
import splitfolders
import constants


class GarbageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=constants.IMAGE_SIZE),
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=constants.IMAGE_SIZE),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.visualisation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=constants.IMAGE_SIZE)
        ])

    def setup(self, stage=None):
        splitfolders.ratio(
            constants.DATA_DIR,
            output=constants.DATA_DIR,
            seed=1337,
            ratio=(.9, .1),
            group_prefix=None
        )
        # build dataset
        self.train_dataset = torchvision.datasets.ImageFolder(constants.TRAIN_DATA_DIR)
        self.sample_dataset = torchvision.datasets.ImageFolder(constants.TRAIN_DATA_DIR)
        self.test_dataset = torchvision.datasets.ImageFolder(constants.TEST_DATA_DIR)

        self.train_dataset.transform = self.augmentation
        self.sample_dataset.transform = self.visualisation_transform
        self.test_dataset.transform = self.transform

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def visual_data_sample(self) -> DataLoader:
        return DataLoader(self.sample_dataset, batch_size=12, shuffle=True)
