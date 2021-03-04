#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2020/9/16
import os
import csv
import numpy as np


class ScenarioPool():

    def __init__(self, cfg):
        self.cfg = cfg
        scenario_pool = []
        if os.path.exists("./scenario_pool/scenario_pool.csv"):
            with open('./scenario_pool/scenario_pool.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    scenario_pool.append(list(map(int, row)))
        self.scenario_pool = np.array(scenario_pool)

    def is_inpool(self, index):
        for scenario in self.scenario_pool:
            intersection = [x for x in scenario if x in index]
            if len(intersection) > (self.cfg.MATCHER.BATCH_NUM/2):
                return False
            else:
                return True
        return True
