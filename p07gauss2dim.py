# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 22/12/2015 18:36:28
# China:������:�й���ѧԺ��ѧ dragonfive
# copyright 1992-2015 dragonfive



import matplotlib.pyplot as plt,numpy as np,random

# �������ɸ�˹�ֲ��Ķ�ά�ռ����ݵ�,�����浽�ļ�gaussData.npy��
def generage_randomdata():
    Sigma = [[1, 0], [0, 1]]
    mu1 = [1, -1];
    # x1, y1 = np.random.multivariate_normal(mu1, Sigma, 200).T
    z1 = np.random.multivariate_normal(mu1, Sigma, 200)

    mu2=[5.5,-4.5];
    z2 = np.random.multivariate_normal(mu2, Sigma, 200)

    mu3=[1,4];
    z3 = np.random.multivariate_normal(mu3, Sigma, 200)

    mu4=[6,4.5];
    z4 = np.random.multivariate_normal(mu4, Sigma, 200)

    mu5=[9,0.0];
    z5 = np.random.multivariate_normal(mu5, Sigma, 200)

    z=np.concatenate((z1,z2,z3,z4,z5),axis=0) # �ϲ�һЩnp������

    # plt.plot(z.T[0],z.T[1],"*")
    # plt.axis('equal');
    # plt.show()

    # �����Ǳ��淽ʽ
    np.savetxt('gaussData.txt',z,fmt=['%s']*z.shape[1],newline='\n');
    # #�����Ƕ�ȡ��ʽ
    # p=np.loadtxt('gaussData.txt');


# ������������ŷʽ����ķ���,�����ǿ��Զ�ά��
def dist_eclud(pointA,pointB):
    return np.sqrt(np.sum(np.power(pointA-pointB,2)))

# ����Ϊ���ݼ���ʼ������
def initial_center(DataSet,num_of_center):
    dims = np.shape(DataSet)[1] # ������ݵ�ά��
    centers=np.mat(np.zeros((num_of_center,dims))) # ����python��matlab��ĺ���;

    # ����ÿ��ά�������ݼ������ֵ����Сֵ,Ȼ���������������ĵ�ĸ�ά�ȵ�ֵ;
    for i in range(0,dims):
        minJ = min(DataSet[:,i])      # �ڶ�ά�����л��ĳһ�еķ���
        rangeJ = float(max(DataSet[:,i])-minJ) # ��仯��Χ,�������������������ѡ����ʹ�õ�����;
        centers[:,i]=minJ+rangeJ*np.random.rand(num_of_center,1) # ����ʹ�õ���np�������random����������������һ��


    return centers;

# ������k��ֵ����Ĺ���
def kMeans(dataSet,num_of_centers):
    # �Ƚ��г�ʼ�����ĵ��ÿ����Ĺ�����
    num_of_point = np.shape(dataSet)[0]
    cluster_of_point=np.mat(np.zeros((num_of_point,2)))  #���ٵ����ɶ�ά����ķ�ʽ;
    cluster_centers = initial_center(dataSet,num_of_centers)
    clusterChanged = True;
    while clusterChanged: # ѭ��ֱ���������仯
        clusterChanged = False;
        # �������¼������ dist_eclud
        for i in range(num_of_point):
            min_dist = np.inf;min_center = -1;
            for j in range(num_of_centers):
                dist_j = dist_eclud(dataSet[i,:],cluster_centers[j,:])
                if dist_j<min_dist:
                    min_dist=dist_j;min_center=j;
            # ���¹���
            if min_center!=cluster_of_point[i,0]:
                clusterChanged = True;
            cluster_of_point[i,:]=min_center,min_dist**2


        # �������ĵ�����
        for i in range(num_of_centers):
            point_in_clusteri = dataSet[ np.nonzero(cluster_of_point[:,0].A==i)[0] ] # ͳ�ƹ���Ϊi������ϴ����ɵ�
            cluster_centers[i,:] = np.mean(point_in_clusteri,axis=0);

    return cluster_centers,cluster_of_point





# generage_randomdata()
# print()
dataset = np.loadtxt('gaussData.txt')
centers , clusters=kMeans(dataset,5)
# print(centers)
# print(clusters)
plt.plot(centers[:,0],centers[:,1],'x')

biaozhi={0.0:'*',1.0:'v',2.0:'^',3.0:'x',4.0:'*'}
for i in range(dataset.shape[0]):
    plt.plot(dataset[i,0],dataset[i,1],biaozhi.get(clusters[i,0]))

plt.show()