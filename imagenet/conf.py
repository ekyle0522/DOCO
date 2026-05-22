# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import math
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Standard'

# Choice of (source, norm, tent)
# - source: baseline without adaptation
# - norm: test-time normalization
# - tent: test-time entropy minimization (ours)
_C.MODEL.ADAPTATION = 'source'
_C.MODEL.THIRD_MODEL_ADAPTATION = 'sar'

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

_C.CORRUPTION.ID_BENCHMARK = "imagenet_c"

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate 
# The 5000 val images defined by Robustbench were actually used:
# Please see https://github.com/RobustBench/robustbench/blob/7af0e34c6b383cd73ea7a1bbced358d7ce6ad22f/robustbench/data/imagenet_test_image_ids.txt
_C.CORRUPTION.NUM_EX = 5000

# Dataset for OOD evaluation
_C.CORRUPTION.OOD_DATASET = 'places365'

# Number of OOD examples to evaluate
_C.CORRUPTION.NUM_OOD_SAMPLES = 5000

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

_C.OPTIM.LR_SCRATCH_NORMAL = 0.1
_C.OPTIM.LR_SCRATCH_AGGRESSIVE = 0.2


_C.OPTIM.DIST_THRESHOLD = 30.0
_C.OPTIM.VARIANCE_THRESHOLD = 0.1

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = False

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1010 #2025 #1 #55 #22

# Output directory
_C.SAVE_DIR = "./output_1k"

# Data directory
_C.DATA_DIR = "/root/data01"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# ViDA parameters
_C.OPTIM.ViDALR = 5e-8
_C.TEST.vida_rank1 = 1
_C.TEST.vida_rank2 = 128
_C.OPTIM.MT_ViDA = 0.999
_C.OPTIM.MT = 0.999

# DPCore parameters
_C.OPTIM.DPCORE_PROMPT_NUM = 8
_C.OPTIM.DPCORE_EMA_ALPHA = 0.999
_C.OPTIM.DPCORE_TEMP_TAU = 3.0
_C.OPTIM.DPCORE_THR_RHO = 0.8


_C.OPTIM.DOCO_BETA = 0.5
_C.OPTIM.DOCO_PROMPT_NUM = 8
_C.OPTIM.DOCO_EMA_ALPHA = 0.999
_C.OPTIM.DOCO_GMM_POOL_SIZE = 512

# EATA + UniEnt parameters
_C.OPTIM.EATA_UNIENT_FISHER_ALPHA = 2000.0
_C.OPTIM.EATA_UNIENT_E_MARGIN = math.log(1000) * 0.40
_C.OPTIM.EATA_UNIENT_D_MARGIN = 0.05
_C.OPTIM.EATA_UNIENT_ALPHA = [1.0, 0.2]
_C.OPTIM.EATA_UNIENT_CRITERION = "ent_unf"


_C.num_classes = 1000

# CDC setting configurations
_C.SRC_NUM_SAMPLES = 300
_C.FISHER_SIZE = 2000

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

#---------------STAMP_config--------------#
_C.STAMP = CfgNode()
_C.STAMP.ALPHA = 0.8
_C.STAMP.NUM_AUG = 15





def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    valid_adaptations = [
        "source",
        "norm",
        "tent",
        "cotta",
        "vida",
        "doco",
        "sarcome",
        "eatacome",
        "ostta",
        "eata",
        "eataunient",
        "stamp",
        "sar",
        "dpcore",
    ]
    assert _C.MODEL.ADAPTATION in valid_adaptations, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    parser.add_argument("--data_dir", default=os.environ.get("DATA_ROOT", '/mnt/d/stamp_lib/datasets'), type=str)
    parser.add_argument("--checkpoint", default='/mnt/d/stamp_lib/ckpt/vida_vit_cifar_ckpt/imagent_vit_vida.pt', type=str)
    parser.add_argument("--unc_thr", default=0.2, type=float)

    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    
    base_name = cfg.MODEL.ADAPTATION

    ood_shorthands = {
        'textures': 'TX',
        'places365': 'PL',
        'inaturalist': 'IN',
        'sun': 'SU',
        'lsun': 'LS',
        'imagenet-o': 'IO',
        'ninco_ood_classes': 'NC',
        'ssb-hard': 'SB'
    }

    def normalize_eval_dataset_name(dataset_name, id_benchmark):
        dataset_name = str(dataset_name).strip().lower()
        id_benchmark = str(id_benchmark).strip().lower()

        if dataset_name in {'imagenet', 'imagenet-c', 'imagenetc', 'imagenet_c'}:
            if id_benchmark in {'laion_c', 'laion-c', 'laionc', 'laion', 'imagenet-l'}:
                return 'laion-c'
            return 'imagenet-c'

        alias_map = {
            'laion-c': 'laion-c',
            'laion_c': 'laion-c',
            'laionc': 'laion-c',
            'imagenet-a': 'imagenet-a',
            'imagenet_a': 'imagenet-a',
            'imagenet-r': 'imagenet-r',
            'imagenet_r': 'imagenet-r',
            'imagenet-sketch': 'imagenet-sketch',
            'imagenet_sketch': 'imagenet-sketch',
        }
        return alias_map.get(dataset_name, dataset_name)

    id_shorthands = {
        'imagenet-c': 'IC',
        'laion-c': 'LC',
        'imagenet-a': 'IA',
        'imagenet-r': 'IR',
        'imagenet-sketch': 'IS',
    }

    normalized_eval_dataset = normalize_eval_dataset_name(
        cfg.CORRUPTION.DATASET,
        getattr(cfg.CORRUPTION, 'ID_BENCHMARK', 'imagenet_c')
    )

    id_dataset_shorthand = id_shorthands.get(normalized_eval_dataset, 'ID')
    ood_dataset_shorthand = ood_shorthands.get(str(cfg.CORRUPTION.OOD_DATASET).lower(), 'OOD')

    
    
    log_name_parts = [base_name, current_time]

    if cfg.CORRUPTION.NUM_OOD_SAMPLES == 0:
        scenario_part = f"closed_{id_dataset_shorthand}"
    else:
        def format_count(n):
            if n == -1: return "All"
            if n >= 1000 and n % 1000 == 0: return f"{n // 1000}k"
            return str(n)

        id_count_str = format_count(cfg.CORRUPTION.NUM_EX)
        ood_count_str = format_count(cfg.CORRUPTION.NUM_OOD_SAMPLES)
        
        scenario_part = (f"{id_dataset_shorthand}{ood_dataset_shorthand}_"
                         f"ID{id_count_str}-OOD{ood_count_str}")

    log_name_parts.append(scenario_part)
    base_log_name = "_".join(log_name_parts)

    hyperparam_part = ""  
    adaptation_method = cfg.MODEL.ADAPTATION

    if adaptation_method == 'dpcore':
        prompt_len = cfg.OPTIM.DPCORE_PROMPT_NUM
        hyperparam_part = f"_L{prompt_len}"
    elif adaptation_method == 'doco':
        prompt_len = cfg.OPTIM.DOCO_PROMPT_NUM
        hyperparam_part = f"_L{prompt_len}"
    elif adaptation_method in ['tent', 'cotta', 'ostta', 'stamp']:
        lrate = cfg.OPTIM.LR
        hyperparam_part = f"_lr{lrate}"
    
    log_dest = f"{base_log_name}{hyperparam_part}.txt"
    

    cfg.DATA_DIR = args.data_dir
    cfg.TEST.ckpt = args.checkpoint


    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)  
    torch.backends.cudnn.deterministic = True
    

    torch.use_deterministic_algorithms(True) 
    
    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)
    return args
