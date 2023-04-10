import torch
import torch.nn as nn

#将通道个数调整为8的整数倍，以便更好的调用硬件设备
#ch输入的卷积核个数，min_ch最小通道数
def _make_divisible(ch , divisor = 8 , min_ch = None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    #采用卷积核最小个数为8
    if min_ch is None:
        min_ch = divisor
    #四舍五入，离ch最近的8的倍数值
    new_ch = max(min_ch , int(ch + divisor / 2) // divisor * divisor)
    #保证向下取整channel不会减少超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return  new_ch

#组合层：卷积层+BN层+Relu6
#Mobile中所有的卷积层和DW层都是这个组合，唯一区别就是倒残差层具有线性激活降维处理
class ConvBNReLU(nn.Sequential):#继承自nn.Sequential
    #groups表示卷积类别，如果为1，表示不同层；如果groups=in_channel则表示DW卷积
    def __init__(self , in_channel , out_channel , kernel_size=3 , stride=1 , groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_channel , out_channel , kernel_size , stride , padding,
                      groups=groups , bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

#倒残差结构：第一层1×1普通卷积层，第二层
class InvertedResidual(nn.Module):
    #expand_ratio扩展因子，即表格中的t
    def __init__(self , in_channel , out_channel , stride , expand_ratio):
        super(InvertedResidual, self).__init__()
        #第一层卷积层
        hidden_channel = in_channel * expand_ratio
        #保证使用short的时候，步长为1且输入维度=输出维度
        self.use_shortcut = stride == 1 and in_channel == out_channel

        #层列表
        layers = []
        #拓展因子不为1，需要添加卷积层
        if expand_ratio != 1:
            #1 × 1 conv
            layers.append(
                ConvBNReLU(in_channel , hidden_channel , kernel_size=1)
            )
        #通过extend添加一些列元素
        layers.extend([
            #3 × 3 conv，DW卷积输入特征矩阵深度=输出特征矩阵
            ConvBNReLU(hidden_channel , hidden_channel , stride=stride , groups=hidden_channel),
            #1×1的普通卷积
            nn.Conv2d(hidden_channel , out_channel , kernel_size=1 , bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self , x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

#MobileNetV2
class MobileNetV2(nn.Module):
    #alpha是V1中使用的超参数，控制倍率
    def __init__(self , num_classes = 1000 , alpha = 1.0 , round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha , round_nearest)
        #最后一层卷积层
        last_channel = _make_divisible(1280 * alpha , round_nearest)

        inverted_residual_setting =[
            #t , c , n , s
            [1 , 16 , 1 , 1],
            [6 , 24 , 2 , 2],
            [6 , 32 , 3 , 2],
            [6 , 64 , 4 , 2],
            [6 , 96 , 3 , 1],
            [6 , 160 , 3 , 2],
            [6 , 320 , 1 , 1],
        ]

        #网络搭建
        features = []
        #conv1 layer
        features.append(ConvBNReLU(3 , input_channel , stride=2))
        #building inverted reverse residual blocks，遍历参数列表，构建逆残差结构
        for t , c , n , s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha , round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel , output_channel ,stride , expand_ratio=t))
                input_channel = output_channel
        #building last sereral layers
        features.append(ConvBNReLU(input_channel,last_channel,1))
        #combine feature layers
        self.features = nn.Sequential(*features)

        #buildeing classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1 , 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel , num_classes)
        )

        #weight initialization
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight , mode = 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m , nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m , nn.Linear):
                nn.init.normal_(m.weight , 0 , 0.01)
                nn.init.zeros_(m.bias)

    def forward(self , x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# from torch import nn
# import torch
#
#
# def _make_divisible(ch, divisor=8, min_ch=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_ch is None:
#         min_ch = divisor
#     new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_ch < 0.9 * ch:
#         new_ch += divisor
#     return new_ch
#
#
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU6(inplace=True)
#         )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, in_channel, out_channel, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         hidden_channel = in_channel * expand_ratio
#         self.use_shortcut = stride == 1 and in_channel == out_channel
#
#         layers = []
#         if expand_ratio != 1:
#             # 1x1 pointwise conv
#             layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
#         layers.extend([
#             # 3x3 depthwise conv
#             ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
#             # 1x1 pointwise conv(linear)
#             nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#         ])
#
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_shortcut:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = _make_divisible(32 * alpha, round_nearest)
#         last_channel = _make_divisible(1280 * alpha, round_nearest)
#
#         inverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         features = []
#         # conv1 layer
#         features.append(ConvBNReLU(3, input_channel, stride=2))
#         # building inverted residual residual blockes
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * alpha, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, last_channel, 1))
#         # combine feature layers
#         self.features = nn.Sequential(*features)
#
#         # building classifier
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(last_channel, num_classes)
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x