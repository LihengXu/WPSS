#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2020/9/9
import numpy as np


class KnnMatcher():

    def __init__(self, cfg):
        self.cfg = cfg
        self.scene_file = self.cfg.MATCHER.SCENE_FILE
        self.batch_num = self.cfg.MATCHER.BATCH_NUM
        self.scene_scape = self.cfg.MATCHER.SCENE_SCOPE
        self.scene_num = len(self.cfg.MATCHER.SCENE_SCOPE)

        galScenario = np.empty((0, self.scene_num))
        galIndex = []
        with open(self.scene_file, 'r') as f:
            for x in f:
                scene_list = x.strip('\n').split(' ')
                encode = np.array([[float(scene_list[x]) for x in self.cfg.MATCHER.SCENE_INDEX]])
                galScenario = np.concatenate((galScenario, encode), axis=0)
                galIndex.append(int(scene_list[0]))

        self.galScenario = galScenario
        self.galIndex = galIndex

    def get_index(self, proScenario):
        diff = self.galScenario - np.array(proScenario)
        nm_diff = diff / self.scene_scape
        sqdiff = nm_diff ** 2
        sqdistance = sqdiff.sum(axis=1)
        distance = sqdistance ** 0.5
        ranker = np.argsort(distance.reshape(-1))
        rs_index = ranker[:self.batch_num]

        # YOLO的结果文件的真值字典的键时编号
        # detectron2的模型的真值字典的键是0开始递增
        # 由于中间会有部分无车数据的去除导致编号不是0开始连续的，所以需要区分处理
        if not self.cfg.MANAGER.DATASET.IS_DETECTRON:
            rs_index = [self.galIndex[x] for x in ranker[:self.batch_num]]

        return rs_index
