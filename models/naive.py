import torch
import pytorch_lightning as pl
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched


class NaiveClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(
            self.hparams.model['backbone'],
            self.hparams.model['num_classes'],
            pretrained=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x).softmax(dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch['labeled']
        ŷ = self.model(x)
        loss = self.criterion(ŷ, y)
        self.train_acc.update(ŷ.softmax(dim=1), y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.train_acc.compute()
        self.log_dict({
            'train/loss': loss,
            'train/acc': acc,
            'step': self.current_epoch,
        })
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ŷ = self.model(x)
        loss = self.criterion(ŷ, y)
        self.valid_acc.update(ŷ.softmax(dim=1), y)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.valid_acc.compute()
        self.log_dict({
            'val/loss': loss,
            'val/acc': acc,
            'step': self.current_epoch,
        })
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        ŷ = self.model(x)
        loss = self.criterion(ŷ, y)
        self.test_acc.update(ŷ.softmax(dim=1), y)
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.test_acc.compute()
        self.log_dict({
            'test/loss': loss,
            'test/acc': acc,
            'step': 0,
        })
        self.test_acc.reset()

    def configure_optimizers(self):
        optim = get_optim(self.parameters(), **self.hparams.optim['optimizer'])
        sched = get_sched(optim, **self.hparams.optim['scheduler'])
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': sched,
                'interval': self.hparams.optim['interval'],
            },
        }
