# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 06/01/2016 13:12:25
# China:������:�й���ѧԺ��ѧ dragonfive 
# any question mail��dragonfive2013@gmail.com
# copyright 1992-2016 dragonfive 

# ����ĺ�������5���Ƽ����Ӧ��4����ɫ
import random as rd
import numpy as np
def getPointAndColor(num):
    points = np.sort(rd.sample(range(1,13)*4,num))[::-1] # ÿ����ֻ����һ��
    colors = [rd.choice('brmf') for i in range(num)]  # ÿ������һ����ɫ
    myDtype = [('point',int),('color',str,1)]
    pAndC = [tuple((points[i],colors[i])) for i in range(num)]
    return np.array(pAndC,myDtype)


# �÷ֹ�����http://news.tongbu.com/44701.html
# 1.�ʼ�ͬ��˳
# 2.ͬ��˳
# 3.����
# 4.��«
# 5.ͬ��
# 6.˳��
# 7.����
# 8.����
# 9.����
# 10.����
def tiaozi(player):
    points = [player[i][0] for i in range(player.size)]
    dui = np.unique(points,return_counts=True)
    # ���ճ��ִ������±�����;ȡ��������ǰ�������±�;
    CountAndIndex=np.array([tuple((dui[0][i],dui[1][i])) for i in range(np.size(dui[1]))],dtype=[('points',int),('count',int)])
    CountAndIndex.sort(order='count')
    # print(CountAndIndex)
    maxDui = CountAndIndex[np.size(dui[1])-1][1] #���ĸ���
    secondDui = CountAndIndex[np.size(dui[1])-2][1] #�δ�ĸ���
    if maxDui == 1: # ����һ��
        return 10
    elif maxDui == 2 and secondDui == 2:#��ʾ������
        return 8
    elif maxDui == 2: #��ʾ��һ��
        return 9
    elif maxDui == 3 and secondDui == 2 : # ��ʾ��«
        return 7
    elif maxDui == 3: #��ʾ3��
        return 3
    elif maxDui == 4: # ��ʾ4��
        return 3

# �ж��ǲ���˳��
def shunzi(player):
    points = np.sort([player[i][0] for i in range(player.size)])[::-1]
    maxp = points[0]
    minp = points[-1]
    if minp==1 and points[-2]==10 and tonghua(player): # �߼�ͬ��˳
        return 1
    elif ((maxp-minp==4) or minp==1 and points[-2]==10)  and tonghua(player): # ͬ��˳
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

# �Ը��Ƶķ�ʽ��������Ƶĵ�����
def getAllpoints(player):
    # �Ȼ�õ����Ӵ�С������,Ȼ�󷵻ض�Ӧ������
    return int(''.join([str(player[i][0]) for i in range(player.size)]))

# �������
def getPukeType(player):
    typeShun = shunzi(player) # ����Ϊ������
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

# ��ö��ӵ�����
def getDuiziSeq(player):
    points = [player[i][0] for i in range(player.size)]
    dui = np.unique(points,return_counts=True)
    # ���ճ��ִ������±�����;ȡ��������ǰ�������±�;
    CountAndIndex=np.array([tuple((dui[0][i],dui[1][i])) for i in range(np.size(dui[1]))],dtype=[('points',int),('count',int)])
    CountAndIndex.sort(order=['count','points'])

    return  int(''.join(map(str,CountAndIndex[:][0])))
#�ȵ���
def comparePoints(player1,player2,type):
    points1 = np.sort([player1[i][0] for i in range(player1.size)])[::-1]
    points2 = np.sort([player2[i][0] for i in range(player2.size)])[::-1]
    if type==2 or type==6: #����˳�ӱȽϴδ�ֵ
        return points2[1]-points1[1] # С��0��1ʤ��
    if type == 5 or type==10:#ͬ�������Ʊ�
        return getAllpoints(player2)-getAllpoints(player1)
    else: # ʣ�µ���һ�� ���� ���� ��«������ �����������͵�������
        seq1 = getDuiziSeq(player1)
        seq2 = getDuiziSeq(player2)
        return seq2-seq1

    # return 0;


# ����Ƚ����˵���
def comparePlayer(player1,plaer2):
    # �ȱ�����
    type1 = getPukeType(player1)
    type2 = getPukeType(plaer2)
    if type1 != type2:
        return type1-type2 # С��0��1ʤ��
    else:
        return comparePoints(player1,plaer2,type2)

if __name__ == '__main__':
    # ���������ҵ���;
    playerA = getPointAndColor(5)
    playerB = getPointAndColor(5)
    print(playerA,playerB,comparePlayer(playerA,playerB))


