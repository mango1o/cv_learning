import torch.functional as F
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self , num_class , initial_weight):
        super(AlexNet, self).__init__()
        #定义卷积神经网络部分，这里用sequential封装，将网络结构打包封装
        self.features = nn.Sequential(
            #padding输入可以为整型int或者元组tuple(n,m)，元组第一位代表上下，第二维代表左右
            #按照左侧补一列，右侧补两列，可以使用nn.ZeroPad2d((1,2,1,2)),左右上下
            #在池化过程中，计算得到非整数，nn会向下取整
            nn.Conv2d(3 , 48 , kernel_size=11 , stride=4 , padding=2),
            #增加计算量，节约内存
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        #AlexNet最后三层全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),#失活节点，而不是删除节点（置零操作
            nn.Linear(128 * 6 * 6 , 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048 , 2048),
            nn.ReLU(inplace=True),
            #num_class表示最后的类别数目
            nn.Linear(2048,num_class),
        )

        if initial_weight:
            self._initialize_weight()

    #定义训练前序传播
    def forward(self,x):
        x = self.features(x)
        #将训练结果展位1维向量
        x = torch.flatten(x , start_dim=1)#也可以使用view展平
        x = self.classifier(x)
        return x

    #初始化权重
    def _initialize_weight(self):
        #self.modules继承nn.module，这样可以遍历每个层结构
        for m in self.modules():
            #判断当前是哪一层结构，是否是卷积层
            if isinstance(m , nn.Conv2d):
                #使用"kaiming"法初始化权重
                nn.init.kaiming_normal_(m.weight , mode='fan_out' , nonlinearity='relu')
                #特别的，如果偏置不为0，则设置为0
                if m.bias is not None:
                    nn.init.constant_(m.bias , 0)
            #如何是全连接层，则使用正态分布初始化
            elif isinstance(m , nn.Linear):
                nn.init.normal_(m.weight , 0 , 0.01)
                nn.init.constant_(m.bias , 0)
    #但实际上pytorch会自动初始化权重以及偏置
    #此处只是提示，如果需要，如何编写代码



