# -*- coding: utf-8  -*-
#__author__ = 'ASUS'

from PIL import Image
from numpy import *
import pylab as plt
import imtools
# im =array(Image.open(r'E:\资料\onedrive\code\test\image\lena.png').convert('L'))
# img2=255-im;
# plt.gray()
# plt.imshow(img2)
# plt.figure()
# plt.gray()
# plt.imshow(im)
# x=range(2,10)
#
# y=[ i**2 for i in x]
# plt.plot(x,y)
# plt.show()
# im =array(Image.open(r'E:\资料\onedrive\code\test\image\lena.png'.decode('utf-8').encode('gbk')).convert('L'))
# im2,cdf=imtools.histeq(im)
# plt.gray()
# plt.subplot(221);plt.imshow(im);plt.title('orgin image')
# plt.subplot(222);plt.hist(im.flatten(),128);plt.title('orgin image hist')
# plt.subplot(223);plt.imshow(im2);plt.title('junheng image')
# plt.subplot(224);plt.hist(im2.flatten(),128);plt.title('junheng image hist')
# plt.show()

#im2=interp(im.flatten(),bins[:-1],cdf)
f=[2,4,6,2,6]
a=[2,4,6]
b=[4,8,16]

d=interp(f,a,b)  #这个插值就是对f中的值如果在a中，就找到对应的在b中的之作为新值;
print d
