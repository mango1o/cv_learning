# DeepLearning-Pytorch

# 项目环境搭建

[pycharm+anacoda搭建pytorch环境](https://blog.csdn.net/weixin_44189155/article/details/126346501)

用pycharm创建项目，基于pytorch

# 实验一：手写数字识别

- 数据集：MNIST数据集识别手写数字

  MNIST包含70000张手写图像：60000张用于训练，10000用于测试。图像是灰度图，像素是28×28，居中

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/c90a616bacce49ee9e81dd413c423d22.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L-d5a2Y55CG5pm6,size_20,color_FFFFFF,t_70,g_se,x_16)

## 设置环境

```python
import torch
import torchvision
from torch.utils.data import DataLoader
```

我们需要用到以上库进行试验

## 思路流程：

- 准备数据，这些需要准备DataLoader
- 构建模型，使用torch构造一个深度神经网络
- 模型训练
- 模型保存以及评估

## 准备训练集和测试集

### torchvision.transforms的图像数据处理方法

1. torchvision.transforms.ToTensor

   把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloatTensor

   其中(H,W,C)意思为(高，宽，通道数)，黑白图片的通道数只有1，其中每个像素点的取值为[0,255],彩色图片的通道数为(R,G,B),每个通道的每个像素点的取值为[0,255]，三个通道的颜色相互叠加，形成了各种颜色

2. torchbision.transforms.Normalize(mean,std)

   给定均值：mean，shape和图片的通道数相同(指的是每个通道的均值)，方差：std，和图片的通道数相同(指的是每个通道的方差)，将会把`Tensor`规范化处理。

   即：`Normalized_image=(image-mean)/std`

3. torchvision.transforms.Compose(transforms)

   组合多个transform

### 准备MNIST数据集的Dataset和DataLoader

```python
train_data = torchvision.datasets.MNIST(
    root="./dataset/",
    train=True,
    transform=torchvision.transforms.Compose({
        torchvision.transforms.ToTensor(),
        #0.1307, .3081是数据集的均值和标准差
        torchvision.transforms.Normalize(
            (0.1307,) , (0.3081,)
        )
    })
)
#准备数据迭代器
train_dataloader = torch.utils.DataLoader(train_data , batch_size = 64 , shuffle = True)

#测试集数据
test_data = torchvision.datasets.MNIST(
    root = "./dataset/",
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307) , (0.3081)
        )
    ])
)
test_dataloader = torch.utils.DataLoader(test_data , batch_size = 64 , shuffle = True)

```

### 构建模型

**全连接层：**当前一层的神经元和前一层的神经元相互链接，其核心操作就是y = wx，即矩阵的乘法，实现对前一层的数据的变换 

构建一个三层的神经网络（两个全连接层和一个输出层）：

- 激活函数的使用

  以Relu激活函数为例，这是包含Relu的库

  ```python
  import torch.nn.functional as F
  ```

- 每一次的数据形状

  原始数据的形状为：[batch_size , 1 , 28 , 28]

  形状我们要修改为：[batch_size,28*28]

  第一个全连接层输出为：[batch_size , 28] (28是人为设定的，可以换成别的数据)

  激活函数不会改变数据的形状

  第二个FC的形状为[batch_size,10]（手写数字0~9，一共10个类别）

- 模型的损失函数

  这是一个多分类问题，对于二分类问题，我们使用sigmoid计算损失函数

  ![img](https://img-blog.csdnimg.cn/0c3677bc6c704536bf33e7aeaa46c4b1.png)

  对于多分类问题，我们不用sigmoid，而是使用softmax计算交叉熵损失

  ```python
  #pytorch中实现方法如下：
  #方法一：
  criterion = nn.CrossEntropyLoss()
  loss = criterion(input,target)
  
  #方法二：
  #1. 对输出值计算softmax和取对数
  output = F.log_softmax(x,dim=-1)
  #2. 使用torch中带权损失
  loss = F.nll_loss(output,target)
  ```

  ![img](https://img-blog.csdnimg.cn/c363df6b28de4e958a9ce80b89cf3015.png)

### 模型训练

1. 实例化模型，设置模型为训练模式
2. 实例化优化器类，实例化损失函数
3. 获取，遍历数据
4. 梯度设置为0
5. 进行前向计算
6. 计算损失
7. 反向传播
8. 更新参数

### 模型的评估

1. 收集损失和准确率，用来计算损失和平均准确率
2. 损失的计算和同训练时损失的计算方法
3. 准确率的计算：
   1. 模型输出形状为[batch_size,10]
   2. 最大值的位置就是预测的目标值（预测值通过进行softmax计算概率，softmax中分母都是相同的，分子越大，概率越呆）
   3. 最大值获取使用toech.max返回最大值和最大值的位置
   4. 返回最大值后，同真实值进行对比([batch_size])，相同则表示预测成功

### **不用进行梯度的计算**

# 实验二：使用pytorch搭建神经网络（LeNet-5

[LeNet-5论文](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

## 网络结构

LetNet-5是一个较为简单的卷积神经网络，如下图所示：

![ff7818be02f52eb088b1b114f664cf2b.png](https://img-blog.csdnimg.cn/img_convert/ff7818be02f52eb088b1b114f664cf2b.png)

- 输入：单通道的二维图像

- 经过两次**卷积层**到**池化层**，在经过**全连接层**，最后到**输出层**

  整体上是：input layer->convulational layer->pooling layer->activation function->convulational layer->pooling layer->activation function->convulational layer->fully connect layer->fully connect layer->output layer.

整个LeNet包括7层：C1,S2,C3,S4,C5,F6,OUTPUT

**参数分析：**

层编号：

- 英文字母+数字
- C——卷积层，S——下采样层（池化），F——全连接层
- 数字代表（总）层数排列下的层数

术语解释：

- 参数——权重w和偏置b
- 连接数——连线数
- 参数计算：每个卷积核对应一个偏置b，卷积核大小对应权重w的个数（注意通道数

### 输入层

输入：32×32像素的图像，通道数为1

### C1层

C1层试卷几层，使用了6个5×5大小的卷积核，padding = 0 , stride = 1进行卷积，得到特征图大小为（32 - 1 + 5 = 28）28×28

参数个数：（5 * 5+1）*6 = 156，因为每个卷积核大小5×5是参数w，1是偏置b

连接数：156 * 28 * 28，每进行一次卷积，都要执行156次运算，一共得到28*28为输出的特征层，故一共进行156 * 28 * 28次卷积

### S2层

S2层是降采样层，使用6个2×2大小的卷积核进行池化，padding = 0 . stride = 2，得到6个14×14大小的特征图![716bd9306e183202277b1373af1486d8.png](https://img-blog.csdnimg.cn/img_convert/716bd9306e183202277b1373af1486d8.png)

S2层其实相当于：降采样+激活。显示降采样，然后激活函数sigmoid非线性输出。先对C1层2×2的视野求和，然后进入激活函数：

![fb91f333c5cbffb5674db828409a0e0f.png](https://img-blog.csdnimg.cn/img_convert/fb91f333c5cbffb5674db828409a0e0f.png)

**参数个数：**（1+1）*6，其中第一个1为池化对应的2 * 2感受野中最大的那个数的权重w，第二个1为偏置

连接数：（2*2 + 1） * 6 * 14 * 14,虽然只选取 2 * 2 感受野之和，但也存在 2*2 的连接数，1 为偏置项的连接，14 *14 为输出特征层，每一个像素都由前面卷积得到，即总共经历 14 *14 次卷积。

### C3层

卷积层，使用16个5 × 5 × n大小的卷积核，padding = 0 ， stride = 1，一共得到16个10×10的特征图：14 - 1 + 5

![441c3b0509734db27072c31b8c97fbf2.png](https://img-blog.csdnimg.cn/img_convert/441c3b0509734db27072c31b8c97fbf2.png)

如上图所示（横坐标C3，纵坐标S2），C3的前六个0 ~ 5特征图，有S2相邻3个特征图作为输入，卷积核尺寸为：5×5×3；C3的6~11，S2相邻4个特征图作为输入，卷积核尺寸：5×5×4；12,13,14特征图由S2间断的4个特征图作为输入：5×5×4；最后一层由S2的所有特征图作为输入，对应卷积核的尺寸为：5×5×6。

![7f3df376b6f0ba12ed66e6078401f5b7.png](https://img-blog.csdnimg.cn/img_convert/7f3df376b6f0ba12ed66e6078401f5b7.png)

**参数个数**：(5 * 5 * 3+1) * 6+(5 * 5 * 4+1) * 6+(5 * 5 * 4+1) * 3+( 5 * 5 * 6+1)*1=1516。

**连接数**：1516 * 10 * 10 = 151600。10*10为输出特征层，每一个像素都由前面卷积得到，即总共经历10*10次卷积。

### S4层

S4层与S2同样为池化层，使用16个2×2大小的卷积核进行池化，padding=0，stride=2，得到16个5×5大小的特征图

参数个数：（1+1）* 16 = 32

连接数：（2 * 2 + 1）*16 * 5 * 5 = 2000

### C5 层

C5 层是卷积层，使用 120 个 5×5x16 大小的卷积核，padding=0，stride=1进行卷积，得到 120 个 1×1 大小的特征图：5-5+1=1。即相当于 120 个神经元的全连接层。

值得注意的是，与C3层不同，这里120个卷积核都与S4的16个通道层进行卷积操作。

参数个数：(5 * 5 * 16+1) * 120=48120。

连接数：48120 * 1 * 1=48120。

### F6层

全连接层，一共有84个神经元，与C5层全连接，即每个神经元都与C5层的120个特征图相连。计算输入向量和权重向量之间的点积，再加上一个偏置，结果通过sigmoid函数输出。

F6 层有 84 个节点，对应于一个 7x12 的比特图，-1 表示白色，1 表示黑色，这样每个符号的比特图的黑白色就对应于一个编码。该层的训练参数和连接数是(120 + 1)x84=10164。

### OUTPUT层

最后的Output层也是全连接层，是Gaussian Conections，采用RBF函数（径向欧式距离函数），计算输入向量和参数向量之间的欧式距离（目前使用sigmoid替代

Output层一共10个节点，代表0~9，假设x是上一次输入，y是RBF的输出，则计算方式是
$$
y_i = ∑_{j=0}^{83}(x_j - w_{ij})^2
\\i ∈[0-9],j∈[0,7*12 - 1]
$$
RBF越接近0，则越接近i，即越接近于 i 的 ASCII 编码图，表示当前网络输入的识别结果是字符 i。

![b2090927f8d7dfd50af5be65c6fe6878.png](https://img-blog.csdnimg.cn/img_convert/b2090927f8d7dfd50af5be65c6fe6878.png)

上图以3为例。

**参数个数**：84*10=840。

**连接数**：84*10=840。

**transformer和vit**

**SIFT和orb**