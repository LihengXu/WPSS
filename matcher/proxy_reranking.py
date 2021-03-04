#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2020/9/9
import numpy as np


class ProxyRerankingMatcher():

    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_num = self.cfg.MATCHER.BATCH_NUM
        self.lamb = self.cfg.MATCHER.RERANKING.LAMB
        self.prerank = np.load(self.cfg.MATCHER.RERANKING.PRERANK)
        self.ins_galFea = np.load(self.cfg.MATCHER.RERANKING.INS_GALFEA)

        assert self.cfg.MATCHER.RERANKING.MODE in ["VGG", "CODE"]
        if self.cfg.MATCHER.RERANKING.MODE == "VGG":
            self.scene_galFea = np.load(self.cfg.MATCHER.RERANKING.PCA_GALFEA)
        elif self.cfg.MATCHER.RERANKING.MODE == "CODE":
            self.scene_galFea = np.load(self.cfg.MATCHER.RERANKING.CC_GALFEA, allow_pickle=True)

        if not self.cfg.MATCHER.RERANKING.HAZE:
            self.ins_galFea = self.ins_galFea[0:5773, :]
            self.scene_galFea = self.scene_galFea[0:5773, :]

        self.galScenario = np.concatenate((self.ins_galFea, self.scene_galFea), axis=1)

    def get_index(self, proScenario):
        proScenario = np.array(proScenario)
        proScenario = np.expand_dims(proScenario, axis=0)
        proScenario1 = proScenario[:, 0:5]
        proScenario2 = proScenario[:, 5:]

        diff1 = self.ins_galFea - proScenario1
        sqdiff1 = diff1 ** 2
        sqdistance1 = sqdiff1.sum(axis=1)
        distance1 = sqdistance1 ** 0.5

        diff2 = self.scene_galFea - proScenario2
        sqdiff2 = diff2 ** 2
        sqdistance2 = sqdiff2.sum(axis=1)
        distance2 = sqdistance2 ** 0.5

        assert distance1.shape == distance2.shape
        distance = (1 - self.lamb) * distance1 + self.lamb * distance2
        index = np.argmin(distance)
        rs_index = self.prerank[index, :self.batch_num].flatten()

        return rs_index
