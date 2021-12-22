#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2021/12/14
import numpy as np


class BaseDiscriminator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha = cfg.DISCRIMINATOR.ALPHA
        self.beta = cfg.DISCRIMINATOR.BETA

        # matcher的部分参数导入
        self.prerank = np.load(self.cfg.MATCHER.RERANKING.PRERANK).astype(np.int16)
        self.ins_galFea = np.load(self.cfg.MATCHER.RERANKING.INS_GALFEA)
        self.dataset_size = self.cfg.MANAGER.DATASET.SIZE
        assert self.cfg.MATCHER.RERANKING.MODE in ["VGG", "CODE"]
        if self.cfg.MATCHER.RERANKING.MODE == "VGG":
            self.scene_galFea = np.load(self.cfg.MATCHER.RERANKING.PCA_GALFEA)
        elif self.cfg.MATCHER.RERANKING.MODE == "CODE":
            self.scene_galFea = np.load(self.cfg.MATCHER.RERANKING.CC_GALFEA, allow_pickle=True)
        if not self.cfg.MATCHER.RERANKING.HAZE:
            self.ins_galFea = self.ins_galFea[0:5773, :]
            self.scene_galFea = self.scene_galFea[0:5773, :]
        self.galScenario = np.concatenate((self.ins_galFea, self.scene_galFea), axis=1)
        self.scenario_length = self.galScenario.shape[1]

    # 计算场景样例集的类心
    def get_centroid(self, rs_list):
        centroid = [0] * self.scenario_length
        for i in rs_list:
            centroid += self.galScenario[i, :]
        centroid = centroid / len(rs_list)
        return centroid

    # 计算场景样例集距类心的平均距离
    def get_cohesion(self, rs_list):
        centroid = self.get_centroid(rs_list)
        dist = 0
        for i in rs_list:
            dist += np.sqrt(np.sum(np.square(self.galScenario[i, :] - centroid)))
        dist /= len(rs_list)
        return dist
    
    # 计算场景样例集的质心距目标距离
    def get_distance(self, rs_list, rs_index):
        centroid = self.get_centroid(rs_list)
        return np.sqrt(np.sum(np.square(rs_index - centroid)))

    # 判别有效命中
    def discriminate_hit(self, rs_list, rs_index):
        # 质心距
        dist1 = get_distance(rs_list, rs_index)
        # 内距
        dist2 = get_cohesion(rs_list)
        
        return dist1 <= self.alpha and dist2 >= self.beta
