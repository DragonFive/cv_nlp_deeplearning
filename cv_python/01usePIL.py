# -*- coding: gbk -*-
#__author__ = 'ASUS'
from PIL import Image
from pylab import *
import os
import re
import chardet
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
# pil_mg=Image.open(r'E:\����\onedrive\code\test\image\lena.png'.decode('utf-8').encode('gbk')).convert('L')
# #im=array(pil_mg.resize((128,128)))   # ����ͼƬ�ķֱ���
# im=array(pil_mg)
# #imshow(im)
# figure()
# #��ʹ����ɫ��Ϣ
# gray()
# # ��ԭ������Ͻ���ʾ����ͼ��
# contour(im,origin='image')
# axis('equal')
# axis('off')
# figure()
# hist(im.flatten(),2)
# # һЩ��
# x=[100,100,400,400]
# y=[200,500,200,500]

# # ʹ�ú�ɫ��״�������
# #plot(x,y,'r*')
#
# # ����������������
# # plot(x[:2],y[:2])
#
# #��ӱ��⣬��ʾ����ͼ��
# title('plotting:"file"')
# show()
path=r'E:\����\onedrive\code\test\image'
files=os.listdir(path)
print [ f for f in files if not zhPattern.search(str(f))]
if zhPattern.search('�Ǻ�'.decode('gbk')):
    print ('�Ǻ�')

print [str(f) for f in files if zhPattern.search(str(f).decode('gbk'))]
for r in [f for f in files if zhPattern.search(str(f).decode('gbk'))]:# ���ڼ���Ƿ��������ַ�
    print chardet.detect(str(r))    # ��������ַ����ı�������


