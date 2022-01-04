import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms

CROP_SIZE = 224
#train_data_dir = "./data_split/train"
train_data_dir = "./data"
val_data_dir = "./data_split/val"
test_data_dir = "./data_split/val"


class GarbageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop((CROP_SIZE, CROP_SIZE)),
                transforms.RandomAffine(degrees=8, translate=(0.07, 0.08),),
                transforms.RandomAutocontrast(),
                #transforms.RandomApply([transforms.ColorJitter(0.1, 0.05, 0.1, 0.1)]),
                transforms.Resize((CROP_SIZE, CROP_SIZE)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(CROP_SIZE, CROP_SIZE)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.num_classes = 8

    def setup(self, stage=None):
        # build dataset
        self.train_dataset = torchvision.datasets.ImageFolder(train_data_dir)
#         self.test_dataset = torchvision.datasets.ImageFolder(test_data_dir)
        self.test_dataset = torchvision.datasets.ImageFolder(train_data_dir)

        self.train_dataset.transform = self.augmentation
        self.test_dataset.transform = self.transform

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )


    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

