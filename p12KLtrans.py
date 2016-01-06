# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 03/01/2016 21:58:25
# China:北京市:科技网 dragonfive
# any question mail：dragonfive1992@gmail.com
# copyright 1992-2015 dragonfive

# 本文件进行K——L变换
import numpy as np

# 下面的函数产生K——L变换的变换矩阵
def K_Ltrans(x):
    # x=np.array([[2,4,5,5,3,2],[2,3,4,5,4,3]])
    y=np.cov(x) # 这里np.cov以行为向量

    # 下面对协方差做特征值分解
    # 下面求特征值和对应的特征向量
    eigval,eigvec=np.linalg.eig(y) # 做特征值分解
    newEig = np.vstack([eigval,eigvec.T]) # 合并两个list

    # 下面是结构化排序的方式，数组中每一个元素是一个tuple 可以模拟结构体
    dtypes=[('val',float),('vec1',float),('vec2',float)]
    newEig2 = np.array(map(tuple,newEig.T), dtype=dtypes)
    sortedlist = np.sort(newEig2,order='val')
    return [ list(tup)[1:3]  for tup in sortedlist ]





