#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2019/11/20
import pickle
import numpy as np
from utils.eval_utils import voc_eval, f1_eval
from utils.misc_utils import read_class_names, AverageMeter


class RewardManager:
    def __init__(self, cfg):

        self.cfg = cfg
        self.is_detectron_flag = self.cfg.MANAGER.DATASET.IS_DETECTRON
        # some path
        self.class_name_path = self.cfg.MANAGER.DATASET.CLASS_NAME_FILE
        self.eval_file_path = self.cfg.MANAGER.DATASET.EVAL_FILE
        self.detectron_gt_path = self.cfg.MANAGER.DATASET.DETECTRON_GT_FILE
        self.detectron_eval_path = self.cfg.MANAGER.DATASET.DETECTRON_EVAL_FILE
        # some numbers
        self.use_voc_07_metric = False
        # some params
        self.classes = read_class_names(self.class_name_path)
        self.class_num = len(self.classes)
        # reward params
        self.clip_rewards = self.cfg.MANAGER.CLIP_REWARDS
        self.beta = self.cfg.MANAGER.ACCURACY_BETA
        self.beta_bias = self.cfg.MANAGER.ACCURACY_BETA
        self.moving_res_f1 = 0.0

    def get_reward(self, action):
        index = action
        dict = {}
        preds = []

        if self.is_detectron_flag:
            with open(self.detectron_gt_path, 'rb') as f:
                gt_dict = pickle.load(f)

            with open(self.detectron_eval_path, 'rb') as f:
                val_preds = pickle.load(f)

        else:
            with open(self.eval_file_path, 'rb') as f:
                gt_dict = pickle.load(f)
                val_preds = pickle.load(f)

        for m in index:
            dict[m] = gt_dict[m]

        for n in val_preds:
            if n[0] in index:
                preds.append(n)

        # compute the F1-score
        npos, nd, rec, prec, f1 = f1_eval(dict, preds, cfg=self.cfg, iou_thres=0.5)
        print('Recall: {:.4f}, Precision: {:.4f}, F1-score: {:.4f}'.format(rec, prec, f1))
        res_f1 = 1 - f1

        # compute the reward
        reward = (res_f1 - self.moving_res_f1)

        # if rewards are clipped, clip them in the range -0.05 to 0.05
        if self.clip_rewards:
            reward = np.clip(reward, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if 0.0 < self.beta < 1.0:
            self.moving_res_f1 = self.beta * self.moving_res_f1 + (1 - self.beta) * res_f1
            self.moving_res_f1 = self.moving_res_f1 / (1 - self.beta_bias)
            self.beta_bias = 0

            reward = np.clip(reward, -0.1, 0.1)

        print()
        print("Manager: EWA res_f1 = ", self.moving_res_f1)

        return reward, res_f1, rec, prec, f1

