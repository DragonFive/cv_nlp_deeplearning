#-*- coding: utf-8 -*-
#__author__ = 'ASUS'

def parzen(x,h,N):
    """
    创建一个方形窗
    :param x: 样本列表，这里每个样本都是用一维特征
    :param h: 立方体边长 这里只有一个特征？
    :param N: 样本总数
    :return: 返回每个样本落入窗内的概率
    """
    f=x[1:N] # 列表切片
    b=0;     # 暂存每次内循环的和
    p= One(1000);
    for fi in f:
        for xj in x:
            if abs((fi-xj)/h)<=1/2: # 判断是否在方窗内部;
                b=b+1;

