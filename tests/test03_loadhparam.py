import pytorch_lightning as pl
m = pl.resume_from_checkpoint("checkpoints/tune_test_MNIST_v0/latest.ckpt")