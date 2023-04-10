import time
from AlexNet import AlexNet

import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms , datasets , utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as opt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    #设定使用CPU或者GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    #对数据进行处理
    data_trainsform = {
        "train" : transforms.Compose(
          [
              transforms.RandomResizedCrop(224),#对图片进行裁剪，从图片中随机获得224大小的区域
              transforms.RandomHorizontalFlip(),#对图像进行随机翻转
              transforms.ToTensor(),
              transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5,0.5,0.5))
          ]
        ),
        "val" : transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    }
    #获取数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd() , "../"))
    # image_path = os.path.join(data_root , "dataset" , "flower_data")
    image_path = os.path.join(data_root, "medical_dataset")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=image_path + "/TRAIN",
                                         transform=data_trainsform["train"])
    train_num = len(train_dataset)
    #获得分类的名称，返回值是一个字典
    #得到(键值，名称)
    flower_list = train_dataset.class_to_idx
    #键值和名称互换，直接通过字典获取类别
    cla_dir = dict((val,key) for key , val in flower_list.items())
    #生成json编码,并保存
    json_str = json.dumps(cla_dir , indent=4)
    with open("class_indices.json" , "w") as json_file:
        json_file.write(json_str)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size = batch_size,
                                         shuffle=True,
                                         num_workers=0)

    #验证集
    val_dataset = datasets.ImageFolder(root=image_path + "/TEST" ,
                                       transform=data_trainsform['val'])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=10000,
                                             shuffle=False,
                                             num_workers=0)
    # #查看数据集
    # test_data_iter = iter(val_loader)
    # test_image , test_label = next(test_data_iter)
    # # show images
    # imshow(utils.make_grid(test_image))

    #定义损失函数，优化器
    net = AlexNet(num_class=4 , initial_weight=True)

    # model_weight_path = './AlexNet_small.pth'
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path , map_location='cpu'))

    net.to(device)#设置运行设备
    loss_function = nn.CrossEntropyLoss()
    #param = list(net.parameters()) #查看参数
    # optimizer = optim.SGD(net.parameters() , lr = 0.01 , momentum=0.9 ,weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters() , lr=0.0002)
    #定义存储路径
    save_path = './AlexNet.pth'
    best_acc = 0.0

    epoches = 12

    for epoch in range(epoches):
        #train
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step , data in enumerate(train_loader , start=0):
            inputs , lables = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = loss_function(outputs , lables.to(device))
            loss.backward()
            optimizer.step()

            #打印训练数据
            running_loss += loss.item()
            print("\rtrain epoch[{}/{}] loss:{: .3f} ".format(epoch+1 , epoches , loss) , end="")
        print("\t time: {:.3f}".format(time.perf_counter() - t1))

        #val
        net.eval()#测试过程中，不需要dropout
        acc = 0.0
        with torch.no_grad():
            for data_val in val_loader:
                val_imgs , val_labs = data_val
                outputs = net(val_imgs.to(device))
                predict = torch.max(outputs , dim=1)[1]
                acc += (predict == val_labs.to(device)).sum().item()
            acc_val = acc / val_num
            if acc_val > best_acc:
                best_acc = acc_val
                torch.save((net.state_dict()) , save_path)
            print('[epoch %d] train_loss : %.3f test_accuracy : %.3f' %
                  (epoch + 1 , running_loss/step , acc / val_num))




def imshow(img):
    #非正则化
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg , (1 , 2, 0)))
    plt.show()

main()
