# -*- coding: utf-8 -*-
#__author__ = 'ASUS'
import os
from PIL import Image
from numpy import *
def get_imlist(path):
    """
    :param path: 路径
    :return:返回以jpg结尾的文件名列表;
    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def imresize(im,sz):
    """
    使用PIL对象重新定义图像数组的大小
    :param im: 图像对应的PIL数组
    :param sz: 改变后的size
    :return:返回改变后的图像的PIL数组
    """
    image=Image.fromarray(uint8(im))
    return array(image.resize(sz))

def histeq(im,nbr_bins=256):
    """
    对一副灰度图像进行直方图均衡化
    :param im:
    :param nbr_bins:
    :return:
    """
    #计算图像的直方图 bins是imhist的长度+1
    imhist,bins=histogram(im.flatten(),nbr_bins,normed=True)
    cdf=imhist.cumsum()#累积分布函数 可以试一试求累加和;
    cdf=255*cdf/cdf[-1];
    #使用累积分布函数的线性插值，计算新的像素值;
    im2=interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape),cdf


def compute_average(imlist):
    """
    计算图像列表的平均图像
    :param imlist: 图像路径列表
    :return:返回的是平均图像的数组
    """
    #打开第一副图像把他保存在浮点型数组中
    averageim=array(Image.open(imlist[0]),'f')
    for imname in imlist[1:]:#从下标为1的位置开始 因为下标为0的已经取过了
        try:
            averageim+=array(Image.open(imname))
        except:
            print imname+'...skipped'
    averageim/=len(imlist)

    #返回uint8型的平均图像
    return array(averageim,'uint8')
