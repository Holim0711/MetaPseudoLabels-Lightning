import os
import sys
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models import *
from datasets import select_datasets
from transforms.cifar10 import train_transform, rand_transform, valid_transform
from transforms.twin import NqTwinTransform


def train(hparams, ckpt_path=None):
    dm = select_datasets(**hparams['dataset'])
    dm.train_transformₗ = NqTwinTransform(train_transform, rand_transform)
    dm.train_transformᵤ = train_transform
    dm.valid_transform = valid_transform

    model = MetaPseudoLabelsClassifier(**hparams)

    logger = TensorBoardLogger("logs", hparams['name'])

    callbacks = [
        ModelCheckpoint(monitor='val/acc', save_last=True),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=callbacks,
        **hparams['trainer'])

    trainer.fit(model, dm)

    trainer.test(datamodule=dm)


def read_hparams():
    with open(sys.argv[2]) as file:
        hparams = json.load(file)
    hparams['name'] = os.path.splitext(os.path.split(sys.argv[2])[1])[0]
    hparams['script'] = ' '.join(sys.argv)
    return hparams


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train(read_hparams())
    elif sys.argv[1] == 'finetune':
        train(read_hparams(), ckpt_path=sys.argv[3])
