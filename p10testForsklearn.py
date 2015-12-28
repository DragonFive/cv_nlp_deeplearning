# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 27/12/2015 22:37:38
# China:北京市:中国科学院大学 dragonfive 
# any question mail：dragonfive1992@gmail.com 
# copyright 1992-2015 dragonfive
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering, KMeans




# 下面用sklearn库做k均值
from p09mySpectral import dist_eclud, get_qinhemat

def sklearn_kmeans():
    dataset = np.loadtxt('gaussData.txt')
    lables = KMeans(5).fit_predict(dataset);

    plt.scatter(dataset[:,0],dataset[:,1],c=lables)
    plt.axis('equal')
    plt.show()

# 下面用sklearn库做谱聚类
def sklearn_spectual():
    samples = np.loadtxt('sample.txt')
    # 建立邻接矩阵
    num_point = np.shape(samples)[0]
    dist_mat = np.mat(np.zeros((num_point,num_point)))
    # 计算每个点与其他所有点的距离
    for index_pointA in range(num_point):
        dist_mat[index_pointA,:]=[dist_eclud(samples[index_pointA],pointB) for pointB in samples]

    qinhedu_mat = get_qinhemat(dist_mat,8,2)

    qinhedu_mat = (qinhedu_mat+qinhedu_mat.T)/2
    lables=spectral_clustering(qinhedu_mat,2)

    clusterA=[i for i in range(0,num_point) if lables[i]==1] #属于第一类的下标集合
    clusterB=[i for i in range(0,num_point) if lables[i]==0]
    plt.plot(samples[clusterA,0],samples[clusterA,1],'cx')
    plt.plot(samples[clusterB,0],samples[clusterB,1],'mo')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    sklearn_spectual();



