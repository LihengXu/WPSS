#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2020/9/8
from yacs.config import CfgNode


def get_defaults_cfg():
    """
    Construct the default configuration tree.

    Returns:
        cfg (CfgNode): the default configuration tree.
    """
    cfg = CfgNode()

    cfg.CSV_SAVE_PATH = "./results/EVB_results.csv"

    cfg["CONTROLLER"] = CfgNode()
    cfg.CONTROLLER.WEIFHTS_DIR_PATH = './weights'
    cfg.CONTROLLER.WEIFHTS_FILE_PATH = 'weights/controller.ckpt'
    cfg.CONTROLLER.NUM_LAYERS = 1             # number of layers of the state space
    cfg.CONTROLLER.MAX_TRIALS = 300           # maximum number of models generated
    cfg.CONTROLLER.EXPLORATION = 0.8          # high exploration for the first 20 steps
    cfg.CONTROLLER.REGULARIZATION = 1e-3      # regularization strength
    cfg.CONTROLLER.CONTROLLER_CELLS = 60      # number of cells in RNN controller
    cfg.CONTROLLER.EMBEDDING_DIM = 20         # dimension of the embeddings for each state
    cfg.CONTROLLER.RESTORE_CONTROLLER = True  # restore controller to continue training
    cfg.CONTROLLER.EXPLORATION_DIS_FACTOR = 0.99

    cfg["MANAGER"] = CfgNode()
    cfg.MANAGER.ACCURACY_BETA = 0.8           # beta value for the moving average of the accuracy
    cfg.MANAGER.CLIP_REWARDS = 0.0            # clip rewards in the [-0.05, 0.05] range
    cfg.MANAGER.DATASET = CfgNode()
    cfg.MANAGER.DATASET.CLASS_NAME_FILE = "./data/EVB/class.names"
    cfg.MANAGER.DATASET.SIZE = 5773
    cfg.MANAGER.DATASET.MODE = "RAMDOM"
    cfg.MANAGER.DATASET.IS_DETECTRON = False
    cfg.MANAGER.DATASET.EVAL_FILE = ""
    cfg.MANAGER.DATASET.DETECTRON_GT_FILE = ""
    cfg.MANAGER.DATASET.DETECTRON_EVAL_FILE = ""
    cfg.MANAGER.CLASSIDX = CfgNode()
    cfg.MANAGER.CLASSIDX.TRUTH = []
    cfg.MANAGER.CLASSIDX.PRED = []

    cfg["MATCHER"] = CfgNode()
    cfg.MATCHER.TYPE = ""
    cfg.MATCHER.SCENE_FILE = ""
    cfg.MATCHER.BATCH_NUM = 5
    cfg.MATCHER.SCENE_SCOPE = []
    cfg.MATCHER.SCENE_INDEX = []
    cfg.MATCHER.RERANKING = CfgNode()
    cfg.MATCHER.RERANKING.MODE = ""
    cfg.MATCHER.RERANKING.HAZE = False
    cfg.MATCHER.RERANKING.LAMB = 0.3
    cfg.MATCHER.RERANKING.PRERANK = ""
    cfg.MATCHER.RERANKING.INS_GALFEA = ""
    cfg.MATCHER.RERANKING.PCA_GALFEA = ""
    cfg.MATCHER.RERANKING.CC_GALFEA = ""
    cfg.MATCHER.SEMI_REPEAT = True

    cfg["DISCRIMINATOR"] = CfgNode()
    cfg.DISCRIMINATOR.TYPE = "BaseDiscriminator"
    cfg.DISCRIMINATOR.ALPHA = 0.8
    cfg.DISCRIMINATOR.BETA = 0.0

    cfg["SEARCHSPACE"] = CfgNode()
    cfg.SEARCHSPACE.NUM_LAYERS = 17
    cfg.SEARCHSPACE.LAYER_NAMES = []
    cfg.SEARCHSPACE.LAYER_VALUES = [[0, 1, 2, 3, 4],
                                    [0, 1, 2],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1],
                                    [0, 1],
                                    [0, 1],
                                    [0, 1],
                                    [0, 1],
                                    [0, 1],
                                    [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
                                    [0, 2, 4, 6],
                                    [0, 5, 10, 15, 20, 25],
                                    [0, 4, 8, 12],
                                    [0, 2, 4, 6],
                                    [0, 2, 5, 7],
                                    [0, 3, 6, 9, 12, 15, 18],
                                    [0, 9, 19, 28, 38, 47, 57],
                                    ]

    cfg["TRAIN"] = CfgNode()
    cfg.TRAIN.HISTORY_STORE = True
    cfg.TRAIN.HISTORY_FILE = 'train_history.csv'
    cfg.TRAIN.BUFFER_FILE = 'buffers.txt'

    cfg["EVAL"] = CfgNode()
    cfg.EVAL.GT_FLAG = True
    cfg.EVAL.RS_FLAG = True

    cfg.EVAL.GT = CfgNode()
    cfg.EVAL.GT.SAVE_PATH = ""

    cfg.EVAL.RS = CfgNode()
    cfg.EVAL.RS.RS_PATH = ""
    cfg.EVAL.RS.SAVE_PATH = ""

    cfg.EVAL.COHESION_RS_PATH = ""

    cfg["SNENARIOPOOL"] = CfgNode()
    cfg.SNENARIOPOOL.FLAG = True

    return cfg


def setup_cfg(cfg, cfg_file, cfg_opts):
    """
    Load a yaml config file and merge it this CfgNode.

    Args:
        cfg (CfgNode): the configuration tree with default structure.
        cfg_file (str): the path for yaml config file which is matched with the CfgNode.
        cfg_opts (list, optional): config (keys, values) in a list (e.g., from command line) into this CfgNode.

    Returns:
        cfg (CfgNode): the configuration tree with settings in the config file.
    """
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(cfg_opts)
    cfg.freeze()

    return cfg
