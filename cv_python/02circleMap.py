# -*- coding: gbk*-
#__author__ = 'ASUS'

from PIL import Image
#from pylab import *
from pylab import array
import matplotlib.pyplot as plt
import os

# im=array(Image.open(r'E:\资料\onedrive\code\test\image\lena.png'))
# plt.imshow(im)
# print 'Please click 3 point'
# x=plt.ginput(4)
# print x
# plt.show()

path=r'E:\资料\onedrive\code\test\image'
imgelist=os.listdir(path)
for image in imgelist:
    im=array(Image.open(path+'\\'+image))
    if len(im.shape)==2:
        plt.gray()
    plt.imshow(im)
    x=plt.ginput(2)
    y=[image]+x
    print y

plt.show()
