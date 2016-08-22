# -*- coding: utf-8 -*-
#__author__ = 'ASUS'

import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread(r'E:\资料\onedrive\code\test\image\lena.png'.decode('utf-8').encode('gbk'),1)
b,g,r = cv2.split(img1)
img= cv2.merge([r,g,b])

# for prow in img:
#     for point in prow:
#         if point==0:
#             point=255
#         else:
#             point=0


# cv2.imshow("grayphto",img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

print(img.shape) #输出图像的长宽和通道数，如果是灰度图，则不输出通道数;
print(img.size)  #输出图像的像素数目
print(img.dtype) #输出图像的数据类型
partail = img[100:140,100:200]
# print px
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(partail)

#px=img[100,100:150]
#print px


plt.show()


