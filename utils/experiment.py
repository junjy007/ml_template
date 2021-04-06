from utils.common import *
from models.cnnclassifier import MNISTClassifier
import pytorch_lightning as pl
from config.config import Config
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import data.mnist_dataset as mnist_ds
from models.cnnclassifier import MNISTClassifier

class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg:Config):
        super(LitDataModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        mnist_ds.prepare_data(self.cfg.data_dir)

    def setup(self, stage=None):
        data_full = mnist_ds.get_trainval_data(self.cfg.data_dir)
        n_trn = int(self.cfg.train_val_split_ratio * len(data_full))
        n_val = len(data_full) - n_trn
        self.data_train, self.data_val = random_split(
            data_full,
            [n_trn, n_val],
            generator=torch.Generator().manual_seed(42))
        
        # steps_per_epoch = len(self.data_train) // self.cfg.batch_size
        # print("steps per epoch", steps_per_epoch)
        # self.cfg.rt_total_steps = steps_per_epoch * self.cfg.max_epoches

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers_train_loader)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers_val_loader)


class LitMNISTClassifier(pl.LightningModule):
    def __init__(self, cfg:Config):
        super(LitMNISTClassifier, self).__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = MNISTClassifier(cfg)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def loss(self, pred, gnd):
        return self.model.cross_entropy_loss(pred, gnd)

    def accuracy(self, logits, labels):
        return self.model.accuracy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss
        #{"train_loss": loss, "train_accuracy": accuracy}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_acc)

    def configure_optimizers(self):
        optms, schds = [], []
        c = self.cfg

        if c.optimiser.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                lr=c.learning_rate,
                                weight_decay=c.weight_decay, 
                                betas=[c.beta_1, c.beta_2], )
            optms.append(optimizer)
        else:
            raise ValueError(f"unrecognised optimiser {c.optimiser}")
        
        return optms, schds

