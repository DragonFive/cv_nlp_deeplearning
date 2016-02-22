# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 22/02/2016 14:28:17
# China:������:�Ƽ��� dragonfive 
# any question mail��dragonfive1992@gmail.com 
# copyright 1992-2016 dragonfive 



# ����������ݼ�����ũ��
from math import log

def calc_ShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    # ����������ֵ�
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # ���������ũ��
    shannonEnt = 0.0
    for key in labelCounts:
        pKey = float(labelCounts[key])/numEntries
        shannonEnt += -1*pKey*log(pKey,2)

    return shannonEnt







