# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 24/01/2016 21:49:28
# China:北京市:科技网 dragonfive 
# any question mail：dragonfive1992@gmail.com 
# copyright 1992-2016 dragonfive

# 将文件记录转换为numpy的解析程序


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator as op


# 对海伦的兴趣进行分类
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = dataSet-np.tile(inX,(dataSetSize,1))
    sqDiffMat=diffMat**2
    distances = sqDiffMat.sum(axis=1)**0.5
    sortedDistIndex = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndex[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=op.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# 从文件中读出数据来
def file2matrix(filename):
    returnMat = np.array(np.loadtxt(filename,usecols={0,1,2}));
    classLabelVector = np.array(np.loadtxt(filename,usecols={3}));
    return returnMat,classLabelVector

# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
# plt.show()



# 对数据集进行归一化
def autoNorm(dataSet):
    # dataSet = np.zeros((1,1))
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(dataSet.shape)
    normDataSet = dataSet - np.tile(minVals,(dataSet.shape[0],1))
    normDataSet = normDataSet/np.tile(ranges,(dataSet.shape[0],1))
    return normDataSet,ranges,minVals



# normMat,ranges,minVals = autoNorm(datingDataMat)
# print(normMat)


# 用测试集进行测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print('分类器分的类是%d,真正的类别是%d'%(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount+=1.0

    print('错误率是%f'%( errorCount/numTestVecs ))

datingClassTest()







