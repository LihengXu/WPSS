#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2020/9/9
import numpy as np
from re_ranking.re_ranking_feature import re_ranking


class ProxyRerankingMatcher():

    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_num = self.cfg.MATCHER.BATCH_NUM
        self.lamb = self.cfg.MATCHER.RERANKING.LAMB
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

    # 获取最邻近的预排序的reranking序列
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

    # 获得最邻近可重复的index
    def get_nearest_repeat(self, semi_list, semi_index):
        temp_list = self.prerank[semi_index, :].flatten()
        for nearest_index in temp_list:
            if nearest_index not in semi_list:
                return nearest_index
        return -1

    # 获得最邻近不可重复的index
    def get_nearest(self, semi_list, rank_list, semi_index):
        temp_list = self.prerank[semi_index, :].flatten()
        for nearest_index in temp_list:
            if (nearest_index not in semi_list) and (nearest_index not in rank_list):
                return nearest_index
        return -1

    # re-ranking计算（预处理）
    def re_rank(self, ins_galFea, pca_galFea, action=[0] * 30, lamb=0.3, nums=100):
        """
        融合instance label和场景特征的距离进行rank
        :param action: scenario list
        :param lamb: rank_param
        :param nums: rank_num
        :return: rank_index
        """
        Fea = np.array(action)
        probFea = np.expand_dims(Fea, axis=0)
        probFea1 = probFea[:, 0:5]
        probFea2 = probFea[:, 5:]

        galFea1 = ins_galFea
        galFea1 = galFea1[0:self.dataset_size, :]
        final_dist1 = re_ranking(probFea1, galFea1, 20, 6, 0.3)
        galFea2 = pca_galFea
        galFea2 = galFea2[0:self.dataset_size, :]
        final_dist2 = re_ranking(probFea2, galFea2, 20, 6, 0.3)

        try:
            assert final_dist1.shape == final_dist2.shape
        except AssertionError as e:
            print('两个距离矩阵shape不匹配')

        final_dist = (1 - lamb) * final_dist1 + lamb * final_dist2
        final_rank = np.argsort(final_dist)

        return final_rank[:, :nums].flatten()
