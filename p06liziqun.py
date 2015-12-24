# -*- coding: gbk -*-
#__author__ = 'dragonfive'
# 22/12/2015 15:56:32
# China:北京市:中国科学院大学 dragonfive 
# copyright 1992-2015 dragonfive


import random,numpy as np
# f(x)=x^3-5x^2-2x+3
def shiyingdu(x):
    return x**3-5*x**2-2*x+3


# 初始化粒子的速度和位置 每个点的局部最优和所有的全局最优
# num为粒子个数 [-2,5]
def initial_lizi(num,lower_bound,upper_bound):
    #下面生成num个随机数;做位置信息
    location=np.array(random.sample(range(lower_bound,upper_bound+1),num))*1.0;# 初始化各个粒子的位置;
    # return location
    # 下面生成速度初始化信息,位置更新的时候如果越界了，注意往回拉
    speed = np.array([0.1]*num);
    # 下面计算初始的局部最优和全局最优;
    localBestY =  [shiyingdu(x) for x in location];
    localBestX = location;
    wholeBestY = max(localBestY);
    wholeBestX = location[localBestY.index(wholeBestY)];
    return location,speed,localBestX,localBestY,wholeBestX,wholeBestY

# 下面更新速度 , 位置，最优值

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



# 下面执行程序

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



