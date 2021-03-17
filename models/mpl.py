import torch
import pytorch_lightning as pl
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched


class MetaPseudoLabelsClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.teacher = get_model(
            self.hparams.model['backbone'],
            self.hparams.model['num_classes'],
            pretrained=self.hparams.model['pretrained'])
        self.student = get_model(
            self.hparams.model['backbone'],
            self.hparams.model['num_classes'],
            pretrained=self.hparams.model['pretrained'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.student(x).softmax(dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            (xᵤ, rᵤ), _ = batch['unlabeled']

            self.teacher.eval()
            with torch.no_grad():
                self.tyᵤ = self.teacher(xᵤ).argmax(dim=1)
            self.teacher.train()

            sŷᵤ = self.student(xᵤ)
            loss = self.criterion(sŷᵤ, self.tyᵤ)
            self.student_lossᵤ = loss.item()
            return {'loss': loss}
        elif optimizer_idx == 1:
            #(xᵤ, rᵤ), _ = batch['unlabeled']
            xₗ, yₗ = batch['labeled']
            """
            self.student.eval()
            with torch.no_grad():
                sŷₗ = self.student(xₗ)
                self.student_lossₗ = self.criterion(sŷₗ, yₗ).item()
                print(sŷₗ.softmax(dim=1))
                self.train_acc.update(sŷₗ.softmax(dim=1), yₗ)
            self.student.train()

            h = self.student_lossᵤ - self.student_lossₗ

            tŷᵤ = self.teacher(xᵤ)
            loss_mpl = h * self.criterion(tŷᵤ, self.tyᵤ)
            """
            tŷₗ = self.teacher(xₗ)
            loss_sup = self.criterion(tŷₗ, yₗ)
            return {'loss': loss_sup}
    """
    def training_epoch_end(self, outputs):
        acc = self.train_acc.compute()
        self.log_dict({
            'train/acc': acc,
            'step': self.current_epoch,
        })
        self.train_acc.reset()
    """

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ŷ = self.student(x)
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
        ŷ = self.student(x)
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
        optim1 = get_optim(self.student.parameters(), **self.hparams.optimizer)
        sched1 = get_sched(optim1, **self.hparams.scheduler)
        optim2 = get_optim(self.teacher.parameters(), **self.hparams.optimizer)
        sched2 = get_sched(optim2, **self.hparams.scheduler)
        return [optim1, optim2], [sched1, sched2]

    def configure_optimizers(self):
        optim1 = get_optim(self.student.parameters(), **self.hparams.optim['student']['optimizer'])
        sched1 = get_sched(optim1, **self.hparams.optim['student']['scheduler'])
        optim2 = get_optim(self.teacher.parameters(), **self.hparams.optim['teacher']['optimizer'])
        sched2 = get_sched(optim2, **self.hparams.optim['teacher']['scheduler'])
        return [
            {
                'optimizer': optim1,
                'lr_scheduler': {
                    'scheduler': sched1,
                    'interval': self.hparams.optim['student']['interval'],
                },
            },
            {
                'optimizer': optim2,
                'lr_scheduler': {
                    'scheduler': sched2,
                    'interval': self.hparams.optim['teacher']['interval'],
                },
            },
        ]
