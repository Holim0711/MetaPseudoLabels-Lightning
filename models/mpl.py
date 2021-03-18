import torch
import pytorch_lightning as pl
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched


class SmoothedCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = torch.nn.functional.one_hot(labels, num_classes=num_classes).float() * (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


class UDACrossEntropy(torch.nn.Module):
    def __init__(self, threshold=0.8, temperature=0.8):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, logits_w, logits_s):
        logits_w = logits_w.detach()
        pseudo_labels = torch.softmax(logits_w / self.temperature, dim=-1)
        masks = torch.max(pseudo_labels, dim=-1)[0].ge(self.threshold).float()
        log_exp = torch.log_softmax(logits_s, dim=-1)
        return -torch.mean((pseudo_labels * log_exp).sum(dim=-1) * masks)


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
        self.criterion = SmoothedCrossEntropy()
        self.uda_criterion = UDACrossEntropy()
        self.teacher_train_acc = pl.metrics.Accuracy()
        self.student_train_acc = pl.metrics.Accuracy()
        self.teacher_valid_acc = pl.metrics.Accuracy()
        self.student_valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.student(x).softmax(dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            (ʷxₗ, ˢxₗ), yₗ = batch['labeled']
            xᵤ, _ = batch['unlabeled']

            self.student.eval()
            with torch.no_grad():
                ˢzₗ = self.student(ʷxₗ)
                ˢloss_old = self.criterion(ˢzₗ, yₗ)
            self.student.train()

            self.teacher.eval()
            with torch.no_grad():
                ᵗỹᵤ = self.teacher(xᵤ).softmax(dim=1)
                ᵗyᵤ = torch.distributions.Categorical(ᵗỹᵤ).sample()
            self.teacher.train()

            ˢzᵤ = self.student(xᵤ)
            loss = self.criterion(ˢzᵤ, ᵗyᵤ)

            self.temp = {
                'pseudo_labels': ᵗyᵤ,
                'student_loss_old': ˢloss_old,
            }
            return {'loss': loss, **self.temp}
        elif optimizer_idx == 1:
            (ʷxₗ, ˢxₗ), yₗ = batch['labeled']
            xᵤ, _ = batch['unlabeled']

            self.student.eval()
            with torch.no_grad():
                ˢʷzₗ = self.student(ʷxₗ)
                ˢloss_new = self.criterion(ˢʷzₗ, yₗ)
            self.student.train()
            self.student_train_acc.update(ˢʷzₗ.softmax(dim=1), yₗ)

            h = self.temp['student_loss_old'] - ˢloss_new

            ᵗzᵤ = self.teacher(xᵤ)
            loss_mpl = h * self.criterion(ᵗzᵤ, self.temp['pseudo_labels'])

            ᵗʷzₗ = self.teacher(ʷxₗ)
            loss_sup = self.criterion(ᵗʷzₗ, yₗ)

            ᵗˢzₗ = self.teacher(ˢxₗ)
            loss_uda = self.uda_criterion(ᵗʷzₗ, ᵗˢzₗ)
            loss_uda *= min(1., self.global_step / self.hparams.UDA['warmup_step'])
            loss_uda *= self.hparams.UDA['factor']

            self.teacher_train_acc.update(ᵗʷzₗ.softmax(dim=1), yₗ)
            return {'loss': loss_mpl + loss_sup + loss_uda}

    def training_epoch_end(self, outputs):
        teacher_loss = torch.stack([x['loss'] for x in outputs[1]]).mean()
        student_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        teacher_acc = self.teacher_train_acc.compute()
        student_acc = self.student_train_acc.compute()
        self.log_dict({
            'train/loss': student_loss,
            'train/acc': student_acc,
            'train/teacher/loss': teacher_loss,
            'train/teacher/acc': teacher_acc,
            'step': self.current_epoch,
        })
        self.teacher_train_acc.reset()
        self.student_train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ᵗz = self.teacher(x)
        ᵗloss = self.criterion(ᵗz, y)
        self.teacher_valid_acc.update(ᵗz.softmax(dim=1), y)
        ˢz = self.student(x)
        ˢloss = self.criterion(ˢz, y)
        self.student_valid_acc.update(ˢz.softmax(dim=1), y)
        return {'teacher_loss': ᵗloss, 'student_loss': ˢloss}

    def validation_epoch_end(self, outputs):
        teacher_loss = torch.stack([x['teacher_loss'] for x in outputs]).mean()
        stduent_loss = torch.stack([x['student_loss'] for x in outputs]).mean()
        teacher_acc = self.teacher_valid_acc.compute()
        studnet_acc = self.student_valid_acc.compute()
        self.log_dict({
            'val/loss': stduent_loss,
            'val/acc': studnet_acc,
            'val/teacher/loss': teacher_loss,
            'val/teacher/acc': teacher_acc,
            'step': self.current_epoch,
        })
        self.teacher_valid_acc.reset()
        self.student_valid_acc.reset()

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
        no_decay = ['bn']
        teacher_parameters = [
            {'params': [p for n, p in self.teacher.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.teacher.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        student_parameters = [
            {'params': [p for n, p in self.student.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.student.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        # TODO: student wait...
        optim1 = get_optim(student_parameters, **self.hparams.optim['student']['optimizer'])
        sched1 = get_sched(optim1, **self.hparams.optim['student']['scheduler'])
        optim2 = get_optim(teacher_parameters, **self.hparams.optim['teacher']['optimizer'])
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
