# -*- coding: utf-8 -*-
#__author__ = 'ASUS'


import numpy
import cv2
import matplotlib.pyplot as plt


# img = cv2.imread(r'E:\资料\onedrive\code\test\image\result.png'.decode('utf8').encode('gbk'),0)
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
# plt.show()

img = cv2.imread(r'E:\资料\onedrive\code\test\image\lena.png'.decode('utf8').encode('gbk'),1)
#I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
region = img[200:370,200:370]
b,g,r = cv2.split(img)

img2 = cv2.merge([r,g,b])
a,b,c = cv2.split(region)
img3 = cv2.merge([c,b,a])
plt.subplot(121);plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.subplot(122);plt.imshow(img3)
#plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
# cv2.imshow("全部".decode('utf8').encode('gbk'),I)
# cv2.imshow("部分".decode('utf8').encode('gbk'),region)
#cv2.waitKey(0)

# cv2.destroyAllWindows()