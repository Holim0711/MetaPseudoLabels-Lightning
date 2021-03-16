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
            self.hparams.model['num_classes'])
        self.student = get_model(
            self.hparams.model['backbone'],
            self.hparams.model['num_classes'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.student(x).softmax(dim=1)

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()

        x, y = batch[0]
        with torch.no_grad():
            probs = self.teacher(x).softmax(dim=1)
            prob_dist = torch.distributions.Categorical(probs)
            y_pseudo = prob_dist.sample()
        ŷ = self.student(x)
        loss = self.criterion(ŷ, y_pseudo)

        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        g1 = [x.grad.new_tensor(device='cpu')
              for x in self.student.parameters()]

        x, y = batch[1]
        self.student.eval()
        ŷ = self.student(x)
        self.student.train()
        loss = self.criterion(ŷ, y)
        self.train_acc.update(ŷ.softmax(dim=1), y)

        opt1.zero_grad()
        self.manual_backward(loss)

        g2 = [x.grad.new_tensor(device='cpu')
              for x in self.student.parameters()]

        h = [(x * y).sum() for x, y in zip(g1, g2)]

        x, y = batch['noisy']
        ŷ = self.teacher(x)
        loss = self.h * self.criterion(ŷ, y_pseudo)

        opt2.zero_grad()
        self.manual_backward(loss)
        opt2.step()

    def training_epoch_end(self, outputs):
        acc = self.train_acc.compute()
        self.log_dict({
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
        optim1 = get_optim(self.student.parameters(), **self.hparams.optimizer)
        sched1 = get_sched(optim1, **self.hparams.scheduler)
        optim2 = get_optim(self.teacher.parameters(), **self.hparams.optimizer)
        sched2 = get_sched(optim2, **self.hparams.scheduler)
        return [optim1, optim2], [sched1, sched2]
