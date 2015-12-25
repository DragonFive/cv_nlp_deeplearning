# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 24/12/2015 11:05:12
# China:北京市:中国科学院大学 dragonfive 
# copyright 1992-2015 dragonfive

#coding=utf-8
#MSC means Multiple Spectral Clustering
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import networkx as nx
import matplotlib.pyplot as plt


def getNormLaplacian(W):
	"""input matrix W=(w_ij)
	"compute D=diag(d1,...dn)
	"and L=D-W
	"and Lbar=D^(-1/2)LD^(-1/2)
	"return Lbar
	"""
	d=[np.sum(row) for row in W]
	D=np.diag(d)
	L=D-W
	#Dn=D^(-1/2)
	Dn=np.power(np.linalg.matrix_power(D,-1),0.5)
	Lbar=np.dot(np.dot(Dn,L),Dn)
	return Lbar

def getKSmallestEigVec(Lbar,k):
	"""input
	"matrix Lbar and k
	"return
	"k smallest eigen values and their corresponding eigen vectors
	"""
	eigval,eigvec=linalg.eig(Lbar)
	dim=len(eigval)

	#查找前k小的eigval
	dictEigval=dict(zip(eigval,range(0,dim)))
	kEig=np.sort(eigval)[0:k]
	ix=[dictEigval[k] for k in kEig]
	return eigval[ix],eigvec[:,ix]

def checkResult(Lbar,eigvec,eigval,k):
	"""
	"input
	"matrix Lbar and k eig values and k eig vectors
	"print norm(Lbar*eigvec[:,i]-lamda[i]*eigvec[:,i])
	"""
	check=[np.dot(Lbar,eigvec[:,i])-eigval[i]*eigvec[:,i] for i in range(0,k)]
	length=[np.linalg.norm(e) for e in check]/np.spacing(1)
	print("Lbar*v-lamda*v are %s*%s" % (length,np.spacing(1)))

g=nx.karate_club_graph()
nodeNum=len(g.nodes())
m=nx.to_numpy_matrix(g)
Lbar=getNormLaplacian(m)
k=2
kEigVal,kEigVec=getKSmallestEigVec(Lbar,k)
print("k eig val are %s" % kEigVal)
print("k eig vec are %s" % kEigVec)
checkResult(Lbar,kEigVec,kEigVal,k)

#跳过k means，用最简单的符号判别的方法来求点的归属

clusterA=[i for i in range(0,nodeNum) if kEigVec[i,1]>0]
clusterB=[i for i in range(0,nodeNum) if kEigVec[i,1]<0]

#draw graph
colList=dict.fromkeys(g.nodes())
for node,score in colList.items():
	if node in clusterA:
		colList[node]=0
	else:
		colList[node]=0.6
plt.figure(figsize=(8,8))
pos=nx.spring_layout(g)
nx.draw_networkx_edges(g,pos,alpha=0.4)
nx.draw_networkx_nodes(g,pos,nodelist=colList.keys(),
		node_color=colList.values(),
		cmap=plt.cm.Reds_r)
nx.draw_networkx_labels(g,pos,font_size=10,font_family='sans-serif')
plt.axis('off')
plt.title("karate_club spectral clustering")
plt.savefig("spectral_clustering_result.png")
plt.show()



