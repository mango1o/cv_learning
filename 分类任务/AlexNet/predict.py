import os
import json

import torch
from PIL import  Image
from torchvision import  transforms
import matplotlib.pyplot as plt

from AlexNet import AlexNet

class_index = ["EOSINOPHIL" , "LYMPHOCYTE" , "MONOCYTE" , "NEUTROPHIL"]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 ,0.5))
        ]
    )

    #load image
    # img_path = "eosinophil.jpeg"
    # img_path = "LYMPHOCYTE.jpeg"
    # img_path = "MONOCYTE.jpeg"
    img_path = "neutrophil.jpeg"
    assert  os.path.exists(img_path) , "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img , dim = 0)

    #create model
    model = AlexNet(num_class=4,initial_weight=True).to(device)

    #load model weights
    weight_path = "./AlexNet.pth"
    assert os.path.exists(weight_path),"file : '{}' does not exist".format(weight_path)
    model.load_state_dict(torch.load((weight_path)))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output , dim=0)
        predict_cla = torch.argmax(predict).numpy()
    # print(predict_cla)
    print_res = "class: {}  prob:{:.3}".format(
        class_index[int(predict_cla)],
        predict[predict_cla].numpy()
    )

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_index[int(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()
