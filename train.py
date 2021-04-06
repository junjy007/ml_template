import argparse
import math
from utils.common import *
from utils.checkpoints import get_checkpoints
from utils.experiment import LitDataModule, LitMNISTClassifier
from config.config import Config
import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from copy import deepcopy

tune_configs = {
    "major0": {
        "learning_rate": tune.loguniform(5e-6, 1e-3),
        "weight_decay": tune.choice([0.0, 1e-3, 1e-2, 0.1]),
        "preset_index": tune.choice([0, 1, 2])
    }
}

def main_tune(base_cfg):
    tune_epoches = base_cfg.hparam_tune_epoches
    tune_name = base_cfg.hparam_tune_name
    assert tune_name in tune_configs.keys(), \
        f"Unknown tune experiment id {tune_name}"
    tune_cfg = tune_configs[tune_name]
    scheduler = ASHAScheduler(
        max_t=tune_epoches,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=list(tune_cfg.keys()), 
        metric_columns=[cfg.hparam_tune_metric,]\
                    + cfg.hparam_tune_more_metrics_report \
                    + ["training_iteration"])

    analysis = tune.run(
        tune.with_parameters(tune_train, base_cfg=base_cfg),
        resources_per_trial={
            "cpu": 1,
            "gpu": base_cfg.hparam_tune_gpu_per_trail,
        },
        metric=base_cfg.hparam_tune_metric,
        mode=base_cfg.hparam_tune_metric_mode,
        config=tune_cfg,
        num_samples=base_cfg.hparam_tune_run_num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=tune_name
    )

    print("Best hyperparameters found were: ", analysis.best_config)

def tune_train(tune_config:dict, base_cfg):
    """
    Wrapper for ray.tune to adjust necessary params
    """
    # note tune config
    cfg = deepcopy(base_cfg)
    for k, v in tune_config.items():
        cfg.__setattr__(k, v)

    trainer = pl.Trainer(
        max_epochs=cfg.hparam_tune_epoches,
        gpus=math.ceil(cfg.hparam_tune_gpu_per_trail),
        logger=pl.loggers.TensorBoardLogger(
            save_dir=tune.get_trial_dir(),
            name=cfg.exp_model_name + '_' + cfg.dataset_name,
            version="."
        ),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                metrics=[cfg.hparam_tune_metric,]\
                    + cfg.hparam_tune_more_metrics_report, 
                on=cfg.hparam_tune_when_report)
       ]
    )
    ldat = LitDataModule(cfg)
    lmod = LitMNISTClassifier(cfg)
    trainer.fit(lmod, ldat)


def main(cfg: Config, **kwargs):
    # TODO-1: to specify starting from check points 
    # TODO-2: to specify the criteria
    print(
        f"Starting experiment {cfg.exp_model_name}, version {cfg.exp_ver_name}",
        f"\nCheckpoints saved to {cfg.checkpoint_dir}")

    checkpoint_callbacks = get_checkpoints(
        experiment_name=cfg.exp_full_name,
        checkpoint_dir=cfg.checkpoint_dir,
        monitor="val_loss",
        mode="min")
    logger = pl.loggers.TensorBoardLogger(cfg.log_dir, name=cfg.exp_full_name)
    pl.seed_everything(cfg.randseed)

    # data and model
    ldat = LitDataModule(cfg)
    lmod = LitMNISTClassifier(cfg)

    if kwargs.get('callbacks'):
        callbacks = checkpoint_callbacks + kwargs['callbacks']
    else:
        callbacks = checkpoint_callbacks

    trainer_args = {
        'default_root_dir': cfg.root_dir,
        'max_epochs': cfg.max_epoches,
        'deterministic': True,
        'logger': logger,
        'callbacks': callbacks,
        'gpus': cfg.gpus
    }
    # if warm start 
    if cfg.resume_from_checkpoint:
        trainer_args['resume_from_checkpoint'] = \
            os.path.join(checkpoint_dir, cfg.resume_from_checkpoint)
        print(f"Resuming from checkpoint {trainer_args['resume_from_checkpoint']}")
    
    trainer = pl.Trainer(**trainer_args)
    print(f"Start training on {cfg.gpus if cfg.gpus else 'cpu'}")
    if not cfg.dry_run:
        trainer.fit(lmod, ldat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Episodic Image Transformer')
    parser = Config.add_argparse_args(parser)
    parser = Config.add_model_specific_args(parser)
    parser = Config.add_tuning_args(parser)
    args = parser.parse_args()
    cfg = Config(args)

    if cfg.hparam_tune_name:
        main_tune(cfg)
    else:
        main(cfg)