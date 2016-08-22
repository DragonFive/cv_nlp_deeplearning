# -*- coding: gbk -*-
#__author__ = 'ASUS'
from PIL import Image
from pylab import *
import os
import re
import chardet
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
# pil_mg=Image.open(r'E:\资料\onedrive\code\test\image\lena.png'.decode('utf-8').encode('gbk')).convert('L')
# #im=array(pil_mg.resize((128,128)))   # 调整图片的分辨率
# im=array(pil_mg)
# #imshow(im)
# figure()
# #不使用颜色信息
# gray()
# # 在原点的左上角显示轮廓图像
# contour(im,origin='image')
# axis('equal')
# axis('off')
# figure()
# hist(im.flatten(),2)
# # 一些点
# x=[100,100,400,400]
# y=[200,500,200,500]

# # 使用红色形状标记描绘点
# #plot(x,y,'r*')
#
# # 描绘链接两个点的线
# # plot(x[:2],y[:2])
#
# #添加标题，显示描绘的图像
# title('plotting:"file"')
# show()
path=r'E:\资料\onedrive\code\test\image'
files=os.listdir(path)
print [ f for f in files if not zhPattern.search(str(f))]
if zhPattern.search('呵呵'.decode('gbk')):
    print ('呵呵')

print [str(f) for f in files if zhPattern.search(str(f).decode('gbk'))]
for r in [f for f in files if zhPattern.search(str(f).decode('gbk'))]:# 用于检测是否有中文字符
    print chardet.detect(str(r))    # 用来检测字符串的编码类型


