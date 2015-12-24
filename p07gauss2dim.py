# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 22/12/2015 18:36:28
# China:北京市:中国科学院大学 dragonfive
# copyright 1992-2015 dragonfive



import matplotlib.pyplot as plt,numpy as np,random

# 用来生成高斯分布的二维空间数据点,并保存到文件gaussData.npy中
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

    z=np.concatenate((z1,z2,z3,z4,z5),axis=0) # 合并一些np的数组

    # plt.plot(z.T[0],z.T[1],"*")
    # plt.axis('equal');
    # plt.show()

    # 下面是保存方式
    np.savetxt('gaussData.txt',z,fmt=['%s']*z.shape[1],newline='\n');
    # #下面是读取方式
    # p=np.loadtxt('gaussData.txt');


# 计算两个向量欧式距离的方法,向量是可以多维的
def dist_eclud(pointA,pointB):
    return np.sqrt(np.sum(np.power(pointA-pointB,2)))

# 下面为数据集初始化中心
def initial_center(DataSet,num_of_center):
    dims = np.shape(DataSet)[1] # 获得数据的维数
    centers=np.mat(np.zeros((num_of_center,dims))) # 发现python与matlab真的好像啊;

    # 对于每个维度求数据集的最大值和最小值,然后随机计算各个中心点的该维度的值;
    for i in range(0,dims):
        minJ = min(DataSet[:,i])      # 在二维数组中获得某一列的方法
        rangeJ = float(max(DataSet[:,i])-minJ) # 求变化范围,这样生成随机数还可以选我们使用的整数;
        centers[:,i]=minJ+rangeJ*np.random.rand(num_of_center,1) # 这里使用的是np库里面的random方法，看起来更快一点


    return centers;

# 下面是k均值聚类的过程
def kMeans(dataSet,num_of_centers):
    # 先进行初始化中心点和每个点的归属簇
    num_of_point = np.shape(dataSet)[0]
    cluster_of_point=np.mat(np.zeros((num_of_point,2)))  #快速地生成二维数组的方式;
    cluster_centers = initial_center(dataSet,num_of_centers)
    clusterChanged = True;
    while clusterChanged: # 循环直到归属不变化
        clusterChanged = False;
        # 下面重新计算距离 dist_eclud
        for i in range(num_of_point):
            min_dist = np.inf;min_center = -1;
            for j in range(num_of_centers):
                dist_j = dist_eclud(dataSet[i,:],cluster_centers[j,:])
                if dist_j<min_dist:
                    min_dist=dist_j;min_center=j;
            # 更新归属
            if min_center!=cluster_of_point[i,0]:
                clusterChanged = True;
            cluster_of_point[i,:]=min_center,min_dist**2


        # 更新中心点坐标
        for i in range(num_of_centers):
            point_in_clusteri = dataSet[ np.nonzero(cluster_of_point[:,0].A==i)[0] ] # 统计归属为i的所有洗标组成的
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