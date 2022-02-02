import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from pytorch_lightning import LightningModule
import torchvision.models as models
import torchmetrics
import constants


class GarbageClassifier(LightningModule):
    def __init__(self, num_classes=9):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(num_filters, num_classes)

        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        self.feature_extractor.eval()

        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        return self.classifier(representations)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=constants.LEARNING_RATE)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.train_accuracy(y_hat, y)

        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.valid_accuracy(y_hat, y)

        self.log("test_accuracy", self.valid_accuracy, on_step=True, on_epoch=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        return loss

    #
    # def validation_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     self.log('train_acc_epoch', self.accuracy)
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}

    # def test_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     logs = {'test_loss': avg_loss}
    #     return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}
