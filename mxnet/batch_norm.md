## 批量归一化的使用

### 网络定义

```{.python .input  n=1}
from mxnet.gluon import nn
def get_net():
    net = nn.Sequential()
    with net.name_scope():
        #first conv
        net.add(nn.Conv2D(20,kernel_size=5))
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation(activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2,strides=2) )
        # second conv
        net.add(nn.Conv2D(50,kernel_size=3))
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation(activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2,strides=2) )
        # fc
        net.add(nn.Flatten())
        net.add(nn.Dense(128))
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation(activation='relu'))
        net.add(nn.Dense(10))
        net.add(nn.BatchNorm(axis=1))
    return net

```

### 数据读取与训练准备


```{.python .input  n=2}
from mxnet import gluon
from mxnet import nd
from mxnet import autograd
import mxnet as mx

import mxnet.ndarray as nd
def load_data_fashion_mnist(batch_size, resize=None, root="./fashion-mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # transform a batch of examples
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x weight x channel to batch x channel x height x weight
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)
def try_gpu():
    try:
        ctx = mx.gpu()
        _=nd.array([0],ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
```

```{.python .input  n=3}
import sys
sys.path.append(".")
import utils
net = get_net()
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
ctx = try_gpu()
print ctx
net.initialize(ctx=ctx)
cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.2})
for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data,label in train_data:
        label=label.as_in_context(ctx)
        with autograd.record():
            output=net(data.as_in_context(ctx))
            loss=cross_entropy_loss(output,label)
        loss.backward()
        trainer.step(batch_size)
        train_loss+=nd.mean(loss).asscalar()
        train_acc+=nd.mean(output.argmax(axis=1)==label).asscalar()
    print("epoch %d. loss = %f , acc = %f."%(epoch,train_loss/len(train_data),train_acc/len(train_data)) )
        
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/dragon/anaconda2/envs/gluon/lib/python2.7/site-packages/mxnet/gluon/data/vision.py:118: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/dragon/anaconda2/envs/gluon/lib/python2.7/site-packages/mxnet/gluon/data/vision.py:122: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "gpu(0)\nepoch 0. loss = 0.507715 , acc = 0.842498.\nepoch 1. loss = 0.324043 , acc = 0.888989.\nepoch 2. loss = 0.275636 , acc = 0.905165.\nepoch 3. loss = 0.247996 , acc = 0.913328.\nepoch 4. loss = 0.224288 , acc = 0.921324.\nepoch 5. loss = 0.206670 , acc = 0.927734.\nepoch 6. loss = 0.193516 , acc = 0.931223.\nepoch 7. loss = 0.179237 , acc = 0.936048.\nepoch 8. loss = 0.167508 , acc = 0.940755.\nepoch 9. loss = 0.156248 , acc = 0.945079.\n"
 }
]
```
