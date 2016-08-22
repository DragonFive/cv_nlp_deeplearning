# -*- coding: utf-8 -*-
# __author__ = 'dragon'


import numpy as np
import cv2
import matplotlib.pyplot as plt

#x = np.arange(0, 5, 0.1);
#y = np.sin(x)
#plt.plot(x, y)

# 'E:\\test\\image\\lena.png'
# .decode('utf8').encode('gbk')
# unicode(r'E:\资料\onedrive\code\test\image\lena.png','unicode')
img=cv2.imread(r'E:\资料\onedrive\code\test\image\lena.png'.decode('utf8').encode('gbk'),1)
cv2.imshow('image',img)
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.imwrite(r'E:\资料\onedrive\code\test\image\lena2.png'.decode('utf8').encode('gbk'),img);

key = cv2.waitKey(0)
if key == 27:
    print '退出'.decode('utf8').encode('gbk')
else:
    print '保存'.decode('utf8').encode('gbk')


#cv2.destroyWindow('image')#可以删除对应的窗口;
#cv2.destroyAllWindows()#删除所有的窗口
