# python_cv_AI_ML
用python做计算机视觉，人工智能，机器学习，模式识别等

06liziqun 用人工智能中的粒子群算法求一个函数的最值
07gauss2dim 用K均值算法进行聚类
08谱聚类参考
09自己写的谱聚类
# k均值聚类
## 算法逻辑
1. 先进行初始化中心点和每个点的归属类
2. 根据每个点与原中心点的距离找到最近的中心点作为归属类
3. 根据每一类的所有点的特征计算平均值来确定新的中心点位置
4. 重复2和3直到所有点的归属都不再变化为止
## python生成随机数据
一维数据用np.random.rand方法
多维数据就用np.random.multivariate_normal
## python保存矩阵到文件
[Numpy提供了几种数据保存的方法](http://www.cnblogs.com/ice-daigua/archive/2012/11/16/2772674.html)
   Numpy提供了几种数据保存的方法。
   以3*4数组a为例：
### 1 tofile
    ``` a.tofile("filename.bin")```
      这种方法只能保存为二进制文件，且不能保存当前数据的行列信息，文件后缀不一定非要是bin，也可以为txt，但不影响保存格式，都是二进制。

这种保存方法对数据读取有要求，需要手动指定读出来的数据的的dtype，如果指定的格式与保存时的不一致，则读出来的就是错误的数据。
       ```b = numpy.fromfile("filename.bin",dtype = **)```
       读出来的数据是一维数组，需要利用
        ```b.shape = 3,4```重新指定维数。
### 2 save

```
numpy.save("filename.npy",a)
```
利用这种方法，保存文件的后缀名字一定会被置为.npy，这种格式最好只用 ```numpy.load("filename")```来读取。
 

###   3.savetxt
```numpy.savetxt("filename.txt",a)

      b =  numpy.loadtxt("filename.txt")
```

用于处理一维和二维数组


## python实现矩阵乘法

np.mat创建矩阵,np.dot进行矩阵点乘，具体可以看相关参考

## 程序实现源代码



# 用谱聚类做数据聚类

## 1. 数据读取
这里读取的结果是一个二维数据点组成的向量
## 2. 根据点的亲和度建立图，并建立邻接矩阵
设点对亲和性（即边权值）采用如下计算公式：
![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-0917fe5d9eaf65bc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
数据图采用 k-近邻方法来生成（即是说，对每个数据点 xi，首先在所有样本中找出不包含 xi 的k个最邻近的样本点，然后xi与每个邻近样本点均有一条边相连，从而完成图构造）。注意，为了保证亲和度矩阵 W 是对称矩阵，可以令 ![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-ff90d057733623b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 其中找k个最紧邻的方法
对每个数据点 xi，首先在所有样本中找出不包含 xi 的k个最邻近的样本点，可以先计算所有点的距离，然后找到距离中的第k小的，比它大的都认为是不相邻吧

## 3. 构建拉普拉斯矩阵并归一化
Lap=D-W
norm_lap=D^(-1/2)*Lap*D^(-1/2)


## 4. 计算k个最小的特征值并计算其对应的特征向量
python实现是求出所有的特征值，并外部排序获得前k小的特征值的下标，从此k个最小的特征值及其对应的特征向量
## 5. 分类
这里有两种方法
### 1.一般做法keans
对特征向量重组，作为特征进行Kmeans聚类
### 2.简单做法
直接比较特征向量里面元素的值

## 程序实现源代码



# 横向对比抽象方法
广义上讲，任何在学习过程中应用到矩阵特征值分解的方法均叫做谱学习方法，比如主成分分析PCA（K-L变换）  谱聚类

谱聚类也可以用来进行图像前景和后景的分割，用的也是颜色距离聚类的方法
其它用颜色距离聚类方法做图像分割的还有K均值聚类，而事实上我们设置阈值分割图像的那个matlab方法，其实上可以看成是二均值聚类
从数据上看，我们都没有进行分类器的训练和评价，所以这些都是非监督的学习方法
