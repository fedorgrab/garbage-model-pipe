import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from pytorch_lightning import LightningModule
import torchvision.models as models
#from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.functional import accuracy

NUM_CLASSES_OLD = 6


class GarbageClassifier(LightningModule):
    def __init__(
        self,
        learning_rate=0.0001,
        input_shape=224,
        num_classes=8,
        pretrained_from_old=True,
    ):
        super().__init__()
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.resnet_model = models.resnet18(pretrained=True)

        for param in self.resnet_model.parameters():
            param.requires_grad = False

        linear_size = list(self.resnet_model.children())[-1].in_features
        self.resnet_model.fc = nn.Linear(linear_size, NUM_CLASSES_OLD)
        self.load_state_dict(torch.load("./model.pt", map_location=torch.device('cpu')))
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

    def forward(self, x):
        return self.resnet_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
