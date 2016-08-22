# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 22/12/2015 15:56:32
# China:������:�й���ѧԺ��ѧ dragonfive 
# copyright 1992-2015 dragonfive


import random,numpy as np
# f(x)=x^3-5x^2-2x+3
def shiyingdu(x):
    return x**3-5*x**2-2*x+3


# ��ʼ�����ӵ��ٶȺ�λ�� ÿ����ľֲ����ź����е�ȫ������
# numΪ���Ӹ��� [-2,5]
def initial_lizi(num,lower_bound,upper_bound):
    #��������num�������;��λ����Ϣ
    location=np.array(random.sample(range(lower_bound,upper_bound+1),num))*1.0;# ��ʼ���������ӵ�λ��;
    # return location
    # ���������ٶȳ�ʼ����Ϣ,λ�ø��µ�ʱ�����Խ���ˣ�ע��������
    speed = np.array([0.1]*num);
    # ��������ʼ�ľֲ����ź�ȫ������;
    localBestY =  [shiyingdu(x) for x in location];
    localBestX = location;
    wholeBestY = max(localBestY);
    wholeBestX = location[localBestY.index(wholeBestY)];
    return location,speed,localBestX,localBestY,wholeBestX,wholeBestY

# ��������ٶ� , λ�ã�����ֵ

def update(location,speed,localBestX,localBestY,wholeBestX,wholeBestY,opr,lower_bound,upper_bound):
    speed=np.array([ speed[i]+0.2*(localBestX[i]-location[i])+0.2*(wholeBestX-location[i])  for i in range(0,len(speed)) ])
    location=np.array([ location[i]+speed[i]  for i in range(0,len(speed))  ])
    for i in range(0,len(location)):
        if location[i]<lower_bound:
            location[i]=lower_bound
        elif location[i] > upper_bound:
            location[i]=upper_bound


    for i in range(0,len(location)):
        if opr*shiyingdu(location[i]) > opr*localBestY[i]:
            localBestY[i] = shiyingdu(location[i])
        else:
            localBestX[i] = location[i];
    if opr == 1 and opr*wholeBestY < max(localBestY):
        wholeBestY = max(localBestY)
        wholeBestX = location[localBestY.index(wholeBestY)]
    elif opr == -1 and opr*wholeBestY > min(localBestY):
        wholeBestY = min(localBestY)
        wholeBestX = location[localBestY.index(wholeBestY)]

    return location,speed,localBestX,localBestY,wholeBestX,wholeBestY;



# ����ִ�г���

location,speed,localBestX,localBestY,wholeBestX,wholeBestY=initial_lizi(5,-2,5)
for i in range(0,100):
    location,speed,localBestX,localBestY,wholeBestX,wholeBestY=update(location,speed,localBestX,localBestY,wholeBestX,wholeBestY,1,-2,5);

print(wholeBestX,wholeBestY);

location,speed,localBestX,localBestY,wholeBestX,wholeBestY=initial_lizi(5,-2,5)
for i in range(0,100):
    location,speed,localBestX,localBestY,wholeBestX,wholeBestY=update(location,speed,localBestX,localBestY,wholeBestX,wholeBestY,-1,-2,5);
print(wholeBestX,wholeBestY);
# print(x,y,z,m,n)

# print(initial_lizi(5,-2,5))



