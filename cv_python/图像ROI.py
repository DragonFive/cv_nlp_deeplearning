# -*- coding: utf-8 -*-
#__author__ = 'ASUS'

import cv2
import numpy as np
import matplotlib.pyplot as mlp




# img = cv2.imread(r'E:\资料\onedrive\code\test\image\ship2.jpg'.decode('utf-8').encode('gbk'),1)
# b,g,r=cv2.split(img);
# img=cv2.merge([r,g,b])
# # 下面的代码把小船只复制几份
# height,width = 265-241,776-743
# smallBoat = img[241:265,743:776]
# img[383:383+height,94:94+width]=smallBoat
# img[383:383+height,130:130+width]=smallBoat
# img[410:410+height,94:94+width]=smallBoat
# img[410:410+height,130:130+width]=smallBoat
#
# img[:,:,0]=255
# mlp.imshow(img)
# mlp.show()

# #下面的程序把图片输出负片
# img = cv2.imread(r'E:\资料\onedrive\code\test\image\result.png'.decode('utf-8').encode('gbk'),0)
# width,height=img.shape
#
# # img2=255-img;#这条语句通过矩阵运算，生成负片
# img2 = cv2.bitwise_not(img) #这样也能生成负片，按位取反
# # print(img.shape)
# # for hehe in img[:,:]==0:
# #     hehe=255
# # for hehe in img[:,:]==255:
# #     hehe=0
#
# cv2.imshow("hehe",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # 下面进行图片混合
# # E:\资料\onedrive\code\test\image#
# img=cv2.imread(r'E:\资料\onedrive\code\test\image\dezert.jpg'.decode('utf-8').encode('gbk'),1)
# b,g,r=cv2.split(img)
# img=cv2.merge([r,g,b])
#
# person =img[213:328,410:487]
# zeroImg = 0*img+person[0,0];
# zeroImg[213:328,330:407]=person
# twoImg = cv2.addWeighted(img,1,zeroImg,0.2,0)
#
# oneImg = 0*img+person[0,0];
# oneImg[213:328,250:327]=person
# threeImg = cv2.addWeighted(twoImg,1,oneImg,0.2,0)
#
# mlp.imshow(threeImg)
# mlp.show()

