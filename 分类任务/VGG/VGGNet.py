import torch
import torch.nn as nn

#VGG
cfgs = {
    'vgg11' : [64 , "M" , 128 , 'M' , 256 , 256 ,'M' ,512 ,512 ,'M' , 512 ,512 ,'M'],
    'vgg13' : [64 , 'M' , 128 , 'M' , 256 , 256 , 'M' , 512 , 512 , 'M' , 512 , 521 ,'M'],
    'vgg16' : [64 , 64 , 'M' , 128 ,128 , 'M' , 256 , 256 ,256 ,'M' ,512 ,512 ,521 ,'M' , 512 ,512 ,512 , 'M'],
    'vgg19' : [64 ,64 , 'M' , 128 ,128 , 'M' , 256 ,256 , 256 ,256 ,'M' ,512 ,512 ,512 ,512 ,'M' , 512 ,512 ,512 ,512 ,'M']
}

#配置卷积层
def make_features(cfg : list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2 , stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels , v , kernel_size=3 , padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
        #非关键字转换
    return nn.Sequential(*layers)

#生成vgg网络
def vgg(model_name = 'vgg11' , **kwargs):
    assert model_name in cfgs ,"Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg) , **kwargs)
    return  model

class VGG(nn.Module):
    def __init__(self , features , num_classes = 1000 , init_weights = False):
        super(VGG, self).__init__()
        self.features = features
        self.classfier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self , x):
        x = self.features(x)
        x = torch.flatten(x , start_dim=1)
        x = self.classfier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias , 0)
            elif isinstance(m , nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

#手写方法，较为复杂
# class VGGNet(nn.Module):
#     def __init__(self , classes_num):
#         super(VGGNet, self).__init__()
#         # 特征提取
#         self.features = nn.Sequential(
#             nn.Conv2d(3 , 64 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64 , 64 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2 , 2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2,2),
#
#             nn.Conv2d(128, 256 , kernel_size=3  ,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256 , 256 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2 , 2),
#
#             nn.Conv2d(256 , 512 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512 , 512 , kernel_size=3 ,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512 , 512 , kernel_size=3 ,padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2 , 2),
#
#             nn.Conv2d(512 , 512 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512 , 512 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512 , 512 , kernel_size=3 , padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2 , 2),
#
#         )
#         # 网络分类
#         self.classfier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(7 * 7 * 512 , 4096),
#             nn.ReLU(inplace=True),
#
#             nn.Dropout(p=0.5),
#             nn.Linear(4096 , 4096),
#             nn.ReLU(inplace=True),
#
#             nn.Dropout(p=0.5),
#             nn.Linear(4096 , classes_num),
#             nn.ReLU(inplace=True),
#
#             nn.Softmax(),
#         )
#
#     def forward(self , x):
#         x = self.classfier(x)
#         x = torch.flatten(x)
#         x = self.features(x)
#         return x