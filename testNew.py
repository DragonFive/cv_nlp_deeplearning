# -*- coding: utf-8 -*-
#__author__ = 'dragon'

from myTools import Getmyip,newPy
from 07gauss2dim import dist_eclud,initial_center,kMeans
import random,numpy as np,networkx as nx



# newPy("m08spectral.py");

# 先计算空间点的欧式距离
def dist_eclud(pointA,pointB):
    return np.sqrt(sum(np.power(pointA-pointB,2)))




# 首先读取原始数据
samples = np.loadtxt('sample.txt')
# print(samples)
# print(np.shape(samples))

# 建立邻接矩阵
num_point = np.shape(samples)[0]
dist_mat = np.mat(np.zeros((num_point,num_point)))
# 计算每个点与其他所有点的距离
for index_pointA in range(num_point):
    dist_mat[index_pointA,:]=[dist_eclud(samples[index_pointA],pointB) for pointB in samples]

# 根据阈值设置返回值，大于阈值为0，小于阈值计算亲和度
def compute_qinhedu(dis,threshold,sigma):
    if dis>threshold or dis==0:
        return 0
    else:
        return np.exp(-1*np.power(dis,2)/(2*np.power(sigma,2)))

#找到每一个行的前k个计算亲和度，其它的赋值为0,返回亲和度矩阵
def get_qinhemat(dist_mat,k,sigma):
    num_point = np.shape(samples)[0]
    qinhedu_mat = np.mat(np.zeros((num_point,num_point)))
    for index_dist in range(num_point):
        # 先找到第k小的距离;先从小到大排序找到第k个值
        temp_vector=dist_mat[index_dist]
        threshold_value = temp_vector.sort()[k]
        # 比阈值大的都设置为0;与自己的亲和度设为0;计算前k个的亲和度
        qinhedu_mat[index_dist]=np.mat([ compute_qinhedu(dist_pointi,threshold_value,sigma) for dist_pointi in dist_mat[index_dist]])

    return qinhedu_mat

# 下面计算归一化的拉普拉斯矩阵
def get_normal_lapalase(qinhedu_mat):
    # 首先获得度向量
    num_point = np.shape(qinhedu_mat)[0]
    du_vec = [sum(dist_vec) for dist_vec in qinhedu_mat]
    du_mat = np.diag(du_vec)
    laplase_mat = dist_mat - qinhedu_mat
    #dn=du_mat^(-1/2)
    dn = np.power(np.linalg.matrix_power(du_mat,-1),0.5) # 里面的求逆？
    normal_lap = np.dot(np.dot(dn,laplase_mat),dn)
    return normal_lap


# 计算矩阵里k个最小的特征值和对应的特征向量
def getKSmallestEigVec(normal_lap,k):
	eigval,eigvec=np.linalg.eig(normal_lap)   #求一个矩阵的特征值和特征向量
	dim=len(eigval)

	#查找前k小的eigval 这里的方法是先生成一个具有比较对象和待找元素的字典，然后对对比较元素排序，从字典中找到前k个对应的序号。
    #np.sort()这个方法不是在本地排序的,而是把排序结果返回
    # zip()函数 返回一个元组的集合 每一个元组从参数中的各个集合按下标序号抽取;
    # 这个字典创建的方式很经典
	dictEigval=dict(zip(eigval,range(0,dim)))
	kEig=np.sort(eigval)[0:k]
	ix=[dictEigval[k] for k in kEig]
	return eigval[ix],eigvec[:,ix]




