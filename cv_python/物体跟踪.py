# -*- coding: utf-8 -*-
"""
__author__ = 'ASUS'
create on 2015-10-25 20:46:29
"""

import cv2
import numpy as np

cap=cv2.VideoCapture(0)
while(1):
    # 获得每一帧
    ret,frame=cap.read()

    # 转换到HSV空间
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # 设定蓝色的阈值
    lower_blue=np.array([100,50,50])
    upper_blue=np.array([130,255,255])

    # 根据阈值构建掩膜
    mask=cv2.inRange(hsv,lower_blue,upper_blue);

    #对原图像和掩膜进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)

    #显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(300)
    if k==27:
        break
    else:
        cv2.destroyWindow('frame')
        cv2.destroyWindow('mask')
        cv2.destroyWindow('res')

cv2.destroyAllWindows()






