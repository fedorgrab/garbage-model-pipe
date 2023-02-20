from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
import splitfolders
import constants
import utils

class GarbageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        augmentation_transforms = [
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=120),
            transforms.RandomSolarize(threshold=195),
            transforms.RandomAffine(
                degrees=50, translate=(0.165, 0.165), scale=(0.45, 1.65)
            ),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.3, saturation=0.35, hue=0.45
            ),
            transforms.RandomErasing(),
        ]
        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                *augmentation_transforms,
                transforms.Resize(size=constants.IMAGE_SIZE),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=constants.IMAGE_SIZE),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.visualisation_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                *augmentation_transforms,
                transforms.Resize(size=constants.IMAGE_SIZE),
            ]
        )

    def setup(self, stage=None):
        utils.download_data()        
        splitfolders.ratio(
            constants.LOCAL_DATA_DIR,
            output=constants.LOCAL_SPLIT_DATA_DIR,
            seed=1337,
            ratio=(0.9, 0.1),
            group_prefix=None,
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
