# -*- coding: utf-8 -*-
#__author__ = 'ASUS'
import cv2

img = cv2.imread(r'E:\资料\onedrive\code\test\image\lena.png'.decode('utf-8').encode('gbk'),1)
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #将真彩图转化为灰度图
cv2.namedWindow("Image")
cv2.imshow("Image", img2)
cv2.waitKey (0)
cv2.destroyAllWindows()