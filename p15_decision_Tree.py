# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 22/02/2016 14:28:17
# China:北京市:科技网 dragonfive 
# any question mail：dragonfive1992@gmail.com 
# copyright 1992-2016 dragonfive 



# 计算给定数据集的香农熵
from math import log

def calc_ShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    # 建立结果的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 下面计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        pKey = float(labelCounts[key])/numEntries
        shannonEnt += -1*pKey*log(pKey,2)

    return shannonEnt







