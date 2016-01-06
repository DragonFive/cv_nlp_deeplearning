# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 06/01/2016 13:12:25
# China:北京市:中国科学院大学 dragonfive 
# any question mail：dragonfive2013@gmail.com
# copyright 1992-2016 dragonfive 

# 下面的函数产生5张牌及其对应的4个花色
import random as rd
import numpy as np
def getPointAndColor(num):
    points = np.sort(rd.sample(range(1,13)*4,num))[::-1] # 每张牌只出现一次
    colors = [rd.choice('brmf') for i in range(num)]  # 每个牌有一个花色
    myDtype = [('point',int),('color',str,1)]
    pAndC = [tuple((points[i],colors[i])) for i in range(num)]
    return np.array(pAndC,myDtype)


# 得分规则是http://news.tongbu.com/44701.html
# 1.皇家同花顺
# 2.同花顺
# 3.四条
# 4.葫芦
# 5.同花
# 6.顺子
# 7.三条
# 8.两对
# 9.对子
# 10.杂牌
def tiaozi(player):
    points = [player[i][0] for i in range(player.size)]
    dui = np.unique(points,return_counts=True)
    # 按照出现次数对下标排序;取出排名在前两个的下标;
    CountAndIndex=np.array([tuple((dui[0][i],dui[1][i])) for i in range(np.size(dui[1]))],dtype=[('points',int),('count',int)])
    CountAndIndex.sort(order='count')
    # print(CountAndIndex)
    maxDui = CountAndIndex[np.size(dui[1])-1][1] #最大的个数
    secondDui = CountAndIndex[np.size(dui[1])-2][1] #次大的个数
    if maxDui == 1: # 不是一对
        return 10
    elif maxDui == 2 and secondDui == 2:#表示是两对
        return 8
    elif maxDui == 2: #表示是一对
        return 9
    elif maxDui == 3 and secondDui == 2 : # 表示葫芦
        return 7
    elif maxDui == 3: #表示3条
        return 3
    elif maxDui == 4: # 表示4条
        return 3

# 判断是不是顺子
def shunzi(player):
    points = np.sort([player[i][0] for i in range(player.size)])[::-1]
    maxp = points[0]
    minp = points[-1]
    if minp==1 and points[-2]==10 and tonghua(player): # 高级同花顺
        return 1
    elif ((maxp-minp==4) or minp==1 and points[-2]==10)  and tonghua(player): # 同花顺
        return 2
    elif(maxp-minp==4) or minp==1 and points[-2]==10:
        return 6
    else:
        return 10

def tonghua(player):
    colors = [player[i][1] for i in range(player.size)]
    if np.unique(colors).size == 1:
        return 5
    else:
        return 10

# 以高牌的方式获得所有牌的点数和
def getAllpoints(player):
    # 先获得点数从大到小的排列,然后返回对应的整数
    return int(''.join([str(player[i][0]) for i in range(player.size)]))

# 获得牌型
def getPukeType(player):
    typeShun = shunzi(player) # 先认为是杂牌
    typeTiao = tiaozi(player)
    typeTong = tonghua(player)

    if typeTiao<=2:
        return typeTiao
    elif typeTiao<=4:
        return typeTiao
    elif typeTong==5:
        return typeTong
    elif typeShun == 6:
        return typeShun
    else:
        return typeTiao

# 获得对子的排序
def getDuiziSeq(player):
    points = [player[i][0] for i in range(player.size)]
    dui = np.unique(points,return_counts=True)
    # 按照出现次数对下标排序;取出排名在前两个的下标;
    CountAndIndex=np.array([tuple((dui[0][i],dui[1][i])) for i in range(np.size(dui[1]))],dtype=[('points',int),('count',int)])
    CountAndIndex.sort(order=['count','points'])

    return  int(''.join(map(str,CountAndIndex[:][0])))
#比点数
def comparePoints(player1,player2,type):
    points1 = np.sort([player1[i][0] for i in range(player1.size)])[::-1]
    points2 = np.sort([player2[i][0] for i in range(player2.size)])[::-1]
    if type==2 or type==6: #两个顺子比较次大值
        return points2[1]-points1[1] # 小于0则1胜出
    if type == 5 or type==10:#同花和杂牌比
        return getAllpoints(player2)-getAllpoints(player1)
    else: # 剩下的是一对 二对 三条 葫芦和四条 做法按对数和点数排序
        seq1 = getDuiziSeq(player1)
        seq2 = getDuiziSeq(player2)
        return seq2-seq1

    # return 0;


# 下面比较两人的牌
def comparePlayer(player1,plaer2):
    # 先比牌型
    type1 = getPukeType(player1)
    type2 = getPukeType(plaer2)
    if type1 != type2:
        return type1-type2 # 小于0则1胜出
    else:
        return comparePoints(player1,plaer2,type2)

if __name__ == '__main__':
    # 获得两个玩家的牌;
    playerA = getPointAndColor(5)
    playerB = getPointAndColor(5)
    print(playerA,playerB,comparePlayer(playerA,playerB))


