"""
Test parse parameters
"""
import os, sys
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
import argparse
from config.config import Config
import pytorch_lightning as pl

class TestC0(pl.LightningModule):
    def __init__(self, cfg:Config):
        super(TestC0, self).__init__()
        self.save_hyperparameters(vars(cfg))
        print(self.hparams)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Config.add_argparse_args(parser)
    parser = Config.add_tuning_args(parser)
    parser = Config.add_model_specific_args(parser)
    args = parser.parse_args()
    cfg = Config(args)

    print("Configuration")
    print(cfg)

    t = TestC0(cfg)
