#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xuliheng time:2020/9/16
from .scenariopool import ScenarioPool


def build_pool(cfg):
    pool = ScenarioPool(cfg)
    return pool
