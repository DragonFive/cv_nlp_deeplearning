# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 03/01/2016 21:58:25
# China:������:�Ƽ��� dragonfive
# any question mail��dragonfive1992@gmail.com
# copyright 1992-2015 dragonfive

# ���ļ�����K����L�任
import numpy as np

# ����ĺ�������K����L�任�ı任����
def K_Ltrans(x):
    # x=np.array([[2,4,5,5,3,2],[2,3,4,5,4,3]])
    y=np.cov(x) # ����np.cov����Ϊ����

    # �����Э����������ֵ�ֽ�
    # ����������ֵ�Ͷ�Ӧ����������
    eigval,eigvec=np.linalg.eig(y) # ������ֵ�ֽ�
    newEig = np.vstack([eigval,eigvec.T]) # �ϲ�����list

    # �����ǽṹ������ķ�ʽ��������ÿһ��Ԫ����һ��tuple ����ģ��ṹ��
    dtypes=[('val',float),('vec1',float),('vec2',float)]
    newEig2 = np.array(map(tuple,newEig.T), dtype=dtypes)
    sortedlist = np.sort(newEig2,order='val')
    return [ list(tup)[1:3]  for tup in sortedlist ]





