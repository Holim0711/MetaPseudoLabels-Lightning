import torch
import pytorch_lightning as pl
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched


class EMAModel(torch.optim.swa_utils.AveragedModel):

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        def ema_fn(p_swa, p_model, n):
            return self.decay * p_swa + (1. - self.decay) * p_model
        super().__init__(model, device, ema_fn)

    def update_parameters(self, model):
        super().update_parameters(model)
        for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
            device = b_swa.device
            b_model_ = b_model.detach().to(device)
            if self.n_averaged == 0:
                b_swa.detach().copy_(b_model_)
            else:
                b_swa.detach().copy_(self.avg_fn(b_swa.detach(), b_model_,
                                                 self.n_averaged.to(device)))


class LabelSmoothedCrossEntropy(torch.nn.Module):

    def __init__(self, ε=0.0, reduction='mean'):
        super().__init__()
        self.ε = ε
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = input.log_softmax(dim=-1)
        weight = input.new_ones(input.size()) * self.ε / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.ε))

        loss = -(weight * log_prob).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class UDACrossEntropy(torch.nn.Module):
    def __init__(self, temperature, threshold, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.reduction = reduction
        self.𝜇ₘₐₛₖ = None

    def forward(self, logits_s, logits_w):
        log_prob = logits_s.log_softmax(dim=-1)
        weight = torch.softmax(logits_w / self.temperature, dim=-1)
        masks = (weight.max(dim=-1)[0] > self.threshold).float()

        loss = -(weight * log_prob).sum(dim=-1) * masks
        self.𝜇ₘₐₛₖ = masks.mean().item()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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
        self.ema = EMAModel(self.student, self.hparams.model['EMA']['decay'])
        self.CE = torch.nn.CrossEntropyLoss()
        self.student_LS_CE = LabelSmoothedCrossEntropy(
            ε=self.hparams.model['label_smoothing']['student'])
        self.teacher_LS_CE = LabelSmoothedCrossEntropy(
            ε=self.hparams.model['label_smoothing']['teacher'])
        self.UDA_CE = UDACrossEntropy(
            temperature=self.hparams.model['UDA']['temperature'],
            threshold=self.hparams.model['UDA']['threshold'])
        self.teacher_train_acc = pl.metrics.Accuracy()
        self.student_train_acc = pl.metrics.Accuracy()
        self.teacher_valid_acc = pl.metrics.Accuracy()
        self.student_valid_acc = pl.metrics.Accuracy()
        self.ema_valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.tmp = {'𝜇ₕ': 0}

    def forward(self, x):
        return self.ema(x).softmax(dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            xₗ, yₗ = batch['labeled']
            (xᵤ, ʳxᵤ), _ = batch['unlabeled']

            self.student.eval()
            with torch.no_grad():
                self.tmp['s_lossₗ'] = self.CE(self.student(xₗ), yₗ)
            self.student.train()

            self.teacher.eval()
            with torch.no_grad():
                ŷᵤ = self.tmp['ŷᵤ'] = self.teacher(xᵤ).argmax(dim=-1)
            self.teacher.train()

            zᵤ = self.student(ʳxᵤ)
            loss = self.student_LS_CE(zᵤ, ŷᵤ)
            self.student_train_acc.update(zᵤ.softmax(dim=1), ŷᵤ)
            return {'loss': loss, **self.tmp}

        elif optimizer_idx == 1:
            xₗ, yₗ = batch['labeled']
            (xᵤ, ʳxᵤ), _ = batch['unlabeled']

            self.student.eval()
            with torch.no_grad():
                self.tmp['ś_lossₗ'] = self.CE(self.student(xₗ), yₗ)
            self.student.train()

            h = self.tmp['s_lossₗ'] - self.tmp['ś_lossₗ']
            self.tmp['𝜇ₕ'] = 0.99 * self.tmp['𝜇ₕ'] + 0.01 * h
            h -= self.tmp['𝜇ₕ']

            λ = self.hparams.model['UDA']['factor']
            λ *= min(1., self.global_step / self.hparams.model['UDA']['warmup'])

            ᵗz = self.teacher(torch.cat((xₗ, xᵤ, ʳxᵤ)))
            ᵗzₗ = ᵗz[:xₗ.shape[0]]
            ᵗzᵤ, ʳzᵤ = ᵗz[xₗ.shape[0]:].chunk(2)
            del ᵗz

            loss_mpl = self.CE(ʳzᵤ, self.tmp['ŷᵤ'])
 
            loss_uda = self.UDA_CE(ʳzᵤ, ᵗzᵤ.clone().detach())
 
            loss_sup = self.teacher_LS_CE(ᵗzₗ, yₗ)
            self.teacher_train_acc.update(ᵗzₗ.softmax(dim=1), yₗ)

            self.log_dict({
                'detail/avg_mpl_signal': self.tmp['𝜇ₕ'],
                'detail/mpl_signal': h,
                'detail/uda_factor': λ,
                'detail/uda_mask': self.UDA_CE.𝜇ₘₐₛₖ,
                'step': self.global_step,
            })
            return {
                'loss': loss_sup + h * loss_mpl + λ * loss_uda,
                'loss_sup': loss_sup,
                'loss_mpl': loss_mpl,
                'loss_uda': loss_uda,
                **self.tmp
            }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        if optimizer_idx == 0:
            self.ema.update_parameters(self.student)

    def training_epoch_end(self, outputs):
        teacher_loss = torch.stack([x['loss'] for x in outputs[1]]).mean()
        student_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        teacher_acc = self.teacher_train_acc.compute()
        student_acc = self.student_train_acc.compute()
        loss_sup = torch.stack([x['loss_sup'] for x in outputs[1]]).mean()
        loss_mpl = torch.stack([x['loss_mpl'] for x in outputs[1]]).mean()
        loss_uda = torch.stack([x['loss_uda'] for x in outputs[1]]).mean()
        self.log_dict({
            'train/student/loss': student_loss,
            'train/student/acc': student_acc,
            'train/teacher/loss': teacher_loss,
            'train/teacher/acc': teacher_acc,
            'detail/loss/sup': loss_sup,
            'detail/loss/mpl': loss_mpl,
            'detail/loss/uda': loss_uda,
            'step': self.current_epoch,
        })
        self.teacher_train_acc.reset()
        self.student_train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ᵗz = self.teacher(x)
        ᵗloss = self.CE(ᵗz, y)
        self.teacher_valid_acc.update(ᵗz.softmax(dim=1), y)
        ˢz = self.student(x)
        ˢloss = self.CE(ˢz, y)
        self.student_valid_acc.update(ˢz.softmax(dim=1), y)
        z = self.ema(x)
        loss = self.CE(z, y)
        self.ema_valid_acc.update(z.softmax(dim=1), y)
        return {'teacher_loss': ᵗloss, 'student_loss': ˢloss, 'ema_loss': loss}

    def validation_epoch_end(self, outputs):
        teacher_loss = torch.stack([x['teacher_loss'] for x in outputs]).mean()
        stduent_loss = torch.stack([x['student_loss'] for x in outputs]).mean()
        ema_loss = torch.stack([x['ema_loss'] for x in outputs]).mean()
        teacher_acc = self.teacher_valid_acc.compute()
        studnet_acc = self.student_valid_acc.compute()
        ema_acc = self.ema_valid_acc.compute()
        self.log_dict({
            'val/loss': ema_loss,
            'val/acc': ema_acc,
            'val/studnet/loss': stduent_loss,
            'val/studnet/acc': studnet_acc,
            'val/teacher/loss': teacher_loss,
            'val/teacher/acc': teacher_acc,
            'step': self.current_epoch,
        })
        self.teacher_valid_acc.reset()
        self.student_valid_acc.reset()
        self.ema_valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        ŷ = self.student(x)
        loss = self.CE(ŷ, y)
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
                nd in n for nd in no_decay)],
             'weight_decay': self.hparams.optim['teacher']['optimizer']['weight_decay']},
            {'params': [p for n, p in self.teacher.named_parameters() if any(
                nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        student_parameters = [
            {'params': [p for n, p in self.student.named_parameters() if not any(
                nd in n for nd in no_decay)],
             'weight_decay': self.hparams.optim['teacher']['optimizer']['weight_decay']},
            {'params': [p for n, p in self.student.named_parameters() if any(
                nd in n for nd in no_decay)],
             'weight_decay': 0.0},
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
                    'name': "lr/student",
                    'scheduler': sched1,
                    'interval': self.hparams.optim['student']['interval'],
                },
            },
            {
                'optimizer': optim2,
                'lr_scheduler': {
                    'name': "lr/teacher",
                    'scheduler': sched2,
                    'interval': self.hparams.optim['teacher']['interval'],
                },
            },
        ]
