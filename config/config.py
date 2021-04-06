import os
import yaml

_presets= {
    0: [64, 64],
    1: [128, 64],
    2: [128, 128],
}

def load_yaml(fn):
    with open(fn, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
            return d
        except yaml.YAMLError as exc:
            print(exc)

class Config:
    default_config = dict(
        randseed=42,
        dry_run=False,
        root_dir=os.getcwd(),
        data_dir=os.path.join(os.path.expanduser("~"), "data", "common"),
        log_dir="logs",
        checkpoint_dir="checkpoints",
        exp_ver_name="v0",
        train_val_split_ratio=0.8,
        num_workers_train_loader=1,
        num_workers_val_loader=1,

        # optimisation
        batch_size=32,
        max_epoches=100,
        learning_rate=1e-4,
        weight_decay=0.0,
        optimiser="Adam",
        beta_1=0.9,
        beta_2=0.999,

        # model parameters
        layer1_size=64,
        layer2_size=64,

        # for tuning
        hparam_tune_epoches=10,
        hparam_tune_gpu_per_trail=1,
        hparam_tune_run_num_samples=10,
        hparam_tune_metric="val_loss",
        hparam_tune_metric_mode="min",
        hparam_tune_when_report="validation_end",
        hparam_tune_more_metrics_report=["train_loss", "train_accuracy"]
    )

    def __init__(self, args):
        """
        args: commandline arguments from argparse, will over-write 
          values in config-file or default settings
        """
        self._yaml_config = self._from_yaml(args.base_config_file)

        # default values applies only of neither specified in commandline
        # or in base config file
        self._set_values(args) 
        self._setup_dirs()
        self._derive_settings()
        self._check_parameters()
    
    @staticmethod    
    def add_argparse_args(parser):
        # - identify the experiment for logging and checkpoint saving
        parser.add_argument('exp_model_name', type=str,
            metavar="EXPERIMENT_MODEL", help="experiment name")
        parser.add_argument('dataset_name', metavar="DATASET",
            type=str, help="dataset name e.g. CIFAR10")
        parser.add_argument('--exp-ver-name', metavar="VER", 
            type=str,
            help="experiment version, default v0")
        # program hyper-params
        parser.add_argument('-n', '--dry-run', action='store_true')
        parser.add_argument('-y', '--base-config-file', type=str, 
            help="YAML exper-specs, can be used as base settings "\
                "with modifications specified here.")
        parser.add_argument('-g', '--gpus', type=int,
            action='extend', nargs="+",
            help="GPUs to use, e.g. -g 0 1 2")
        parser.add_argument('--root-dir', type=str,
            help="dir of logs and checkpoints default: PWD")
        parser.add_argument('--data-dir', type=str, 
            help="dir contains the dataset, e.g. CIFAR10/, see --dataset-name")
        parser.add_argument('--log-dir', type=str, 
            help="root-log dir, loggers to manage sub-dirs of experiments "
                "default $root_dir/logs")
        parser.add_argument('--checkpoint-dir', type=str, 
            help="root-checkpoint dir, "
            "logs will be saved to .../full-exp-name/ "
            "default: $root_dir/checkpoints")
        parser.add_argument('-c', '--resume-from-checkpoint', type=str, 
            metavar='WARM-START-CHECKPOINT',
            help='FULL PATH to checkpoint file, will supersede all other settings')

        # high-level training process hyper-params
        parser.add_argument('--randseed', type=int,
            help="random seed, default 42")
        parser.add_argument('--train-val-split-ratio', type=float, 
            help="train in (train + val) split")
        parser.add_argument('--num-workers-train-loader', type=int,
            help="#. train loader workers")
        parser.add_argument('--num-workers-val-loader', type=int,
            help="#. train loader workers")
        return parser

    @staticmethod
    def add_tuning_args(parent_parser):
        parent_parser.add_argument('--hparam-tune-name', 
            type=str,
            help="If set, will perform the corresponding hyper-parameter "
                 "tuning. See tuning-functions in experiment.py")
        parent_parser.add_argument('--hparam-tune-epoches', 
            type=int,
            help="epoches in tuning rounds")
        parent_parser.add_argument('--hparam-tune-gpu-per-trail', 
            type=float,
            help="A fractional number, eg. 0.25, specifies how a GPU is shared "
            "among multiple tuning trails. All tasks sharing the GPU start "
            "simutaneously. They should be able to fit in GPU Mem together.")
        parent_parser.add_argument('--hparam-tune-run-num-samples', 
            type=int,
            help="num_samples in ray[tune], random start in grid-search x N")
        parent_parser.add_argument('--hparam-tune-metric', 
            type=str,
            help="metric to tune the hparams, e.g. 'val_loss'")
        parent_parser.add_argument('--hparam-tune-metric-mode', 
            type=str,
            help="which is better, min/max")
        parent_parser.add_argument('--hparam-tune-when-report', 
            type=str,
            help="when to report to tuner: e.g. 'validation_end'")
        parent_parser.add_argument('--hparam-tune-more-metrics-report',
            type=str,
            action="extend", nargs='+',
            help="what's more to report, e.g. train_loss train_accuracy"
        )
        return parent_parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        pp = parent_parser
        pp.add_argument('--batch-size', type=int, help="samples in a batch")
        pp.add_argument('--max-epoches', type=int, help="training epoches")
        pp.add_argument('--learning-rate', type=float, help="lr")
        pp.add_argument('--lr-schedule', type=str, 
            help="learning rate scheduler default None, if specified "\
                "there will be scheduler specific arguments")
        pp.add_argument('--weight-decay', type=float, 
            help="weight decay")
        pp.add_argument('-o', '--optimiser', type=str,
            help="optimiser")

        pp.add_argument('--model-dim', type=int, 
            help="top-level transformer model dimension")

        pg = pp.add_argument_group("Network structure setting - specific")
        pg.add_argument('--layer1-size', type=int, 
            help="neurons in layer 1")
        pg.add_argument('--layer2-size', type=int, 
            help="neurons in layer 2")

        pg = pp.add_argument_group("Network structure setting - presets")
        hstr = "\n".join([f"Index {k}: {v}" 
            for k, v in _presets.items()])
        pg.add_argument('--preset-index', type=int, 
            help=hstr)

        pg = pp.add_argument_group("Adam Optimizer")
        pg.add_argument('--beta-1', type=float, help="adam-b1")
        pg.add_argument('--beta-2', type=float, help="adam-b2")

        return pp

    def __str__(self):
        s0 = \
            f"{self.exp_full_name}:"
        s1 = "\n".join([
            f"\t{k}:{v}" for k, v in vars(self).items()
        ])
        return s0 + s1

    def _set_values(self, args):
        print("== Set configurable hyper-parameters ==")
        # first pass cli
        for a, v in vars(args).items():
            fv = self._yaml_config.get(a)
            if v is not None:
                print(f"Setting {a} to cli-assigned value {v}")
                self.__setattr__(a, v)
                if fv:
                    print(f"\tOverwriting config-file value {fv}")
                
                if a == 'preset_index':
                    for _k in ["layer1_size", "layer2_size"]:
                        try:
                            del Config.default_config[_k]
                        except:
                            pass
                        try:
                            del self._yaml_config[_k]
                        except:
                            pass

        # second pass config file
        for a, v in vars(args).items():
            fv = self._yaml_config.get(a)
            if v is None and fv is not None:
                print(f"Setting {a} to config-file value {fv}")
                self.__setattr__(a, fv)

                if a == 'preset_index':
                    for _k in ["layer1_size", "layer2_size"]:
                        try:
                            del Config.default_config[_k]
                        except:
                            pass

        # 3rd pass, default
        for a, v in vars(args).items():
            fv = self._yaml_config.get(a)
            if v is not None or fv is not None:
                continue
            if a in Config.default_config.keys():
                dv = Config.default_config[a]
                print(f"Setting {a} to default value {dv}")
                self.__setattr__(a, dv)
            else:
                print(f"None valid value found for {a}")
                self.__setattr__(a, None)

        self.dataset_name = self.dataset_name.upper()

    def _setup_dirs(self):
        osp = os.path
        full_pth = lambda s: osp.abspath(osp.expanduser(s))
        if not osp.isabs(self.root_dir):
            self.root_dir = full_pth(self.root_dir)

        if not osp.isabs(self.data_dir):
            self.data_dir = osp.join(self.root_dir, self.data_dir)
        # print(self.data_dir, self.dataset_name)
        # self.data_dir = osp.join(self.data_dir, self.dataset_name)

        if not osp.isabs(self.log_dir):
            self.log_dir = osp.join(self.root_dir, self.log_dir)

        if not osp.isabs(self.checkpoint_dir):
            self.checkpoint_dir = osp.join(self.root_dir, self.checkpoint_dir)

        print("== Directories ==")
        print(f"         data:{self.data_dir}")
        print(f"          log:{self.log_dir}")
        print(f"  checkpoints:{self.checkpoint_dir}")

    def _from_yaml(self, yaml_fname:str) -> dict:
        if yaml_fname:
            if os.path.isabs(yaml_fname):
                self.base_config_file = yaml_fname
            else:
                self.base_config_file = os.path.abspath(
                    os.path.expanduser(yaml_fname))

            print(f"Loading base settings from {self.base_config_file}")
            d = load_yaml(yaml_fname)
        else:
            d = {}
        return d

    def _derive_settings(self):
        if self.dataset_name.upper() == "MNIST":
            self.cx, self.cy = 28, 28
            self.num_classes = 10

        is_layers_set = self.layer1_size or self.layer2_size
        if is_layers_set and self.preset_index is not None:
            raise ValueError("preset-index and layer-sizes "\
                "should not be both set.")
        if self.preset_index is not None:
            p = _presets[self.preset_index]
            self.layer1_size = p[0]
            self.layer2_size = p[1]

        # book-keeping
        self.exp_full_name = '_'.join([
            self.exp_model_name, 
            self.dataset_name, 
            self.exp_ver_name])
        self.checkpoint_dir = os.path.join(
            self.checkpoint_dir, self.exp_full_name)

    def _check_parameters(self):
        pass
        

class EasyConfig:
    """
    similar to pl.Trainer's parameter parser, but in a much simpler way and
    only handle used parameters
    """
    # Default settings and typical values
    # individual, see init
    randseed = 42

    # data
    data_dir = os.path.expanduser("~/data/common")
    num_classes = 10
    train_val_split_ratio = 0.8
    num_workers_train_loader = 6
    num_workers_val_loader = 6

    # optimisation
    batch_size = 32
    max_epoches = 999
    learning_rate = 1e-5
    weight_decay = 0

    lr_schedule = None
    optimiser = "Adam"
    beta_1 = 0.9
    beta_2 = 0.999

    # model
    obj_weight_move = 0.1
    max_episode_steps = 4
    does_sample_random_action = False

    cy, cx = 32, 32 # one example grid setting
    patch_rows, patch_cols = 4, 4
    patch_cy, patch_cx = cy // patch_rows, cx // patch_cols
    num_patches = patch_rows * patch_cols
    patch_content_dim = patch_cy * patch_cx * 3

    # Model Params
    # - global
    model_dim = 256
    num_heads = 8
    num_enc_layers = 6
    max_epsode_steps = 6
    # - local attention model 
    model_dim_2 = 128
    num_heads_2 = 4
    num_enc_layers_2 = 4
    agg_zoom_patch_dim = model_dim_2

    # - Local
    patch_rows_2, patch_cols_2 = 8, 8
    patch_cy_2, patch_cx_2 = patch_cy // patch_rows_2, patch_cx // patch_cols_2
    num_patches_2 = patch_rows_2 * patch_cols_2
    patch_dim_2 = patch_cy_2 * patch_cx_2 * 3

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.k = v
    
    def __str__(self):
        s1 = "\n".join([
            f"\t{k}:{v}" for k, v in vars(self).items()
        ])
        return s1

