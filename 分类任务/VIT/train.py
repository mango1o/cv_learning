import torch
import torch.nn as nn
from torchvision import transforms , datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import time
import numpy as np
import torchvision.models.resnet
from ViT_model import vit_base_patch16_224_in21k as create_model

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
              transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229,0.224,0.225])
          ]
        ),
        "val" : transforms.Compose(
            [
                transforms.Resize((256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485 , 0.456 , 0.406], [0.229 , 0.224 , 0.225])
            ]
        )
    }
    #获取数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd() , "../"))
    image_path = os.path.join(data_root , "dataset" , "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=image_path + "/train",
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
    val_dataset = datasets.ImageFolder(root=image_path + "/val" ,
                                       transform=data_trainsform['val'])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)
    #查看数据集
    # test_data_iter = iter(val_loader)
    # test_image , test_label = next(test_data_iter)
    # # show images
    # imshow(utils.make_grid(test_image))

    #定义损失函数，优化器
    net = create_model(num_classes=5, has_logits=False).to(device)

    ####载入预训练模型
    # model_weight_path = './resnet34_pre.pth'
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path , map_location='cpu'))
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel , 5)

    ####

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    #param = list(net.parameters()) #查看参数
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5E-5)
    #定义存储路径
    save_path = './vit.pth'
    best_acc = 0.0

    epoches = 4

    train_steps = len(train_loader)
    for epoch in range(epoches):
        #train
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        # train_bar = tqdm(train_loader, file=sys.stdout)
        for step , data in enumerate(train_loader , start=0):
            inputs , lables = data
            optimizer.zero_grad()
            logits = net(inputs.to(device))
            loss = loss_function(logits , lables.to(device))
            loss.backward()
            optimizer.step()

            #打印训练数据
            running_loss += loss.item()
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epoches,
            #                                                          loss)
            rate = (step + 1) / len(train_loader)
            print("\rtrain epoch[{}/{}] loss:{: .3f} ".format(epoch + 1, epoches, loss), end="")
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
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, acc_val))




def imshow(img):
    #非正则化
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg , (1 , 2, 0)))
    plt.show()

main()

# import os
# import math
# import argparse
#
# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# import torch.optim.lr_scheduler as lr_scheduler
#
# from ShuffleNetV2 import shufflenet_v2_x1_0
# from my_dataset import MyDataSet
# from utils import read_split_data, train_one_epoch, evaluate
#
#
# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     print(args)
#     print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
#     tb_writer = SummaryWriter()
#     if os.path.exists("./weights") is False:
#         os.makedirs("./weights")
#
#     train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
#
#     data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#         "val": transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#
#     # 实例化训练数据集
#     train_dataset = MyDataSet(images_path=train_images_path,
#                               images_class=train_images_label,
#                               transform=data_transform["train"])
#
#     # 实例化验证数据集
#     val_dataset = MyDataSet(images_path=val_images_path,
#                             images_class=val_images_label,
#                             transform=data_transform["val"])
#
#     batch_size = args.batch_size
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw,
#                                                collate_fn=train_dataset.collate_fn)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=nw,
#                                              collate_fn=val_dataset.collate_fn)
#
#     # 如果存在预训练权重则载入
#     model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
#     if args.weights != "":
#         if os.path.exists(args.weights):
#             weights_dict = torch.load(args.weights, map_location=device)
#             load_weights_dict = {k: v for k, v in weights_dict.items()
#                                  if model.state_dict()[k].numel() == v.numel()}
#             print(model.load_state_dict(load_weights_dict, strict=False))
#         else:
#             raise FileNotFoundError("not found weights file: {}".format(args.weights))
#
#     # 是否冻结权重
#     if args.freeze_layers:
#         for name, para in model.named_parameters():
#             # 除最后的全连接层外，其他权重全部冻结
#             if "fc" not in name:
#                 para.requires_grad_(False)
#
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#
#     for epoch in range(args.epochs):
#         # train
#         mean_loss = train_one_epoch(model=model,
#                                     optimizer=optimizer,
#                                     data_loader=train_loader,
#                                     device=device,
#                                     epoch=epoch)
#
#         scheduler.step()
#
#         # validate
#         acc = evaluate(model=model,
#                        data_loader=val_loader,
#                        device=device)
#
#         print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
#         tags = ["loss", "accuracy", "learning_rate"]
#         tb_writer.add_scalar(tags[0], mean_loss, epoch)
#         tb_writer.add_scalar(tags[1], acc, epoch)
#         tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
#
#         torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_classes', type=int, default=5)
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--lrf', type=float, default=0.1)
#
#     # 数据集所在根目录
#     # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
#     parser.add_argument('--data-path', type=str,
#                         default="./dataset/flower_data/flower_photos")
#
#     # shufflenetv2_x1.0 官方权重下载地址
#     # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
#     parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
#                         help='initial weights path')
#     parser.add_argument('--freeze-layers', type=bool, default=False)
#     parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
#
#     opt = parser.parse_args()
#
#     main(opt)