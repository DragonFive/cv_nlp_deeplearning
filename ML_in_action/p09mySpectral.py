# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 25/12/2015 10:57:07
# China:������:�Ƽ��� dragonfive
# any question mail��dragonfive1992@gmail.com
# copyright 1992-2015 dragonfive 

from p07gauss2dim import initial_center,kMeans
import random,numpy as np,networkx as nx,matplotlib.pyplot as plt

# �ȼ���ռ���ŷʽ����
def dist_eclud(pointA,pointB):
    return np.sqrt(sum(np.power(pointA-pointB,2)))

# ������ֵ���÷���ֵ��������ֵΪ0��С����ֵ�����׺Ͷ�
def compute_qinhedu(dis,threshold,sigma):
    if dis>threshold or dis==0:
        return 0
    else:
        return np.exp(-1*np.power(dis,2)/(2*np.power(sigma,2)))

#�ҵ�ÿһ���е�ǰk�������׺Ͷȣ������ĸ�ֵΪ0,�����׺ͶȾ���
def get_qinhemat(dist_mat,k,sigma):
    num_point = np.shape(dist_mat)[0]
    qinhedu_mat = np.mat(np.zeros((num_point,num_point)))
    for index_dist in range(num_point):
        # ���ҵ���kС�ľ���;�ȴ�С���������ҵ���k��ֵ
        temp_vector=dist_mat[index_dist,:]
        threshold_value = np.sort(temp_vector)[0,k]
        # ����ֵ��Ķ�����Ϊ0;���Լ����׺Ͷ���Ϊ0;����ǰk�����׺Ͷ�
        qinhedu_mat[index_dist]=np.mat([ compute_qinhedu(dist_mat[index_dist,dist_pointi],threshold_value,sigma) for dist_pointi in range(num_point)])

    return qinhedu_mat

# ��������һ����������˹����
def get_normal_lapalase(qinhedu_mat):
    # ���Ȼ�ö�����
    num_point = np.shape(qinhedu_mat)[0]
    du_vec = [np.sum(dist_vec) for dist_vec in qinhedu_mat]
    du_mat = np.diag(du_vec)
    laplase_mat = dist_mat - qinhedu_mat
    #dn=du_mat^(-1/2)
    dn = np.power(np.linalg.matrix_power(du_mat,-1),0.5) # ��������棿dist_mat
    normal_lap = np.dot(np.dot(dn,laplase_mat),dn)
    return normal_lap


# ���������k����С������ֵ�Ͷ�Ӧ����������
def getKSmallestEigVec(normal_lap,k):
	eigval,eigvec=np.linalg.eig(normal_lap)   #��һ�����������ֵ����������
	dim=len(eigval)

	#����ǰkС��eigval ����ķ�����������һ�����бȽ϶���ʹ���Ԫ�ص��ֵ䣬Ȼ��ԶԱȽ�Ԫ�����򣬴��ֵ����ҵ�ǰk����Ӧ����š�
    #np.sort()������������ڱ��������,���ǰ�����������
    # zip()���� ����һ��Ԫ��ļ��� ÿһ��Ԫ��Ӳ����еĸ������ϰ��±���ų�ȡ;
    # ����ֵ䴴���ķ�ʽ�ܾ���
	dictEigval=dict(zip(eigval,range(0,dim)))
	kEig=np.sort(eigval)[0:k]
	ix=[dictEigval[k] for k in kEig]
	return eigval[ix],eigvec[:,ix]   # ���������ǰ��д�ŵ�

if __name__=="__main__":

    # ���ȶ�ȡԭʼ����
    samples = np.loadtxt('sample.txt')
    # print(samples)
    # print(np.shape(samples))

    # �����ڽӾ���
    num_point = np.shape(samples)[0]
    dist_mat = np.mat(np.zeros((num_point,num_point)))
    # ����ÿ�������������е�ľ���
    for index_pointA in range(num_point):
        dist_mat[index_pointA,:]=[dist_eclud(samples[index_pointA],pointB) for pointB in samples]
    # print(dist_mat[1:6,1:6])
    # dist_mat = (dist_mat+dist_mat.T)/2
    #
    # print(dist_mat[1:6,1:6])
    qinhedu_mat = get_qinhemat(dist_mat,70,35)

    qinhedu_mat = (qinhedu_mat+qinhedu_mat.T)/2

    norm_lap = get_normal_lapalase(qinhedu_mat)
    keigval,keigvec=getKSmallestEigVec(norm_lap,4)

    centers , clusters=kMeans(keigvec,2)


    biaozhi={0.0:'cx--',1.0:'mo:'}
    for i in range(samples.shape[0]):
        plt.plot(samples[i,0],samples[i,1],biaozhi.get(clusters[i,0]))

    plt.show()


    # clusterA=[i for i in range(0,num_point) if keigvec[i,0]>0] #���ڵ�һ����±꼯��
    # clusterB=[i for i in range(0,num_point) if keigvec[i,0]<0]
    # print(keigvec[:,0])
    # print(keigvec[:,1])
    #
    # plt.plot(samples[clusterA,0],samples[clusterA,1],'*')
    # plt.plot(samples[clusterB,0],samples[clusterB,1],'x')
    # plt.axis('equal')
    # plt.show()
