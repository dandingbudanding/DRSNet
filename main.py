#encoding:utf-8
import torch
from torchvision.transforms import transforms as T
import time
import cv2
import PIL.Image as Image
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from thop import profile
import torch.nn.functional as F
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080

import models.my_unet as my_unet
import models.enet as enet
import models.segnet as segnet
import models.AODNet as aodnet
import models.my_CascadeNet as my_cascadenet
import models.my_DRSNet as my_drsnet
import models.resnext_unet as resnext_unet
import models.unet_nest as unet_nest
import models.resnet34_unet as unet_res34
import models.trangle_net as mytrangle_net
from my_metric import SegmentationMetric
from my_dataloader import LiverDataset,LiverDataset_three,weighing
from torch.optim import lr_scheduler
import  models.resnet50_unet as resnet50_unet
import models.dfanet as dfanet
import models.lednet as lednet
import models.CGNet  as cgnet
import models.PSPNet as pspnet
import models.BiSeNet as bisenet
import models.ESPNet as espnet
import models.FDDWNet as fddwnet
import models.ContextNet as contextnet
import models.LinkNet as linknet
import models.EDANet as edanet
import models.ERFNet as erfnet

from models.losses import focal_loss
from models.losses.losses import LovaszLossSoftmax,LovaszLossHinge


# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
# x_transform = T.ToTensor()
y_transform = T.ToTensor()


def train_model(model, criterion, optimizer, dataload, lr_scheduler):
    num_epochs=args.num_epochs
    loss_record=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)  # 前向传播
            outputs=outputs.squeeze()
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            # print(lr_scheduler.get_lr()[0])
            optimizer.step()
            lr_scheduler.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            loss_record.append(loss.item())
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_data = pd.DataFrame( data=loss_record)
    loss_data.to_csv(args.loss_record)
    plt.plot(loss_data)
    torch.save(model.state_dict(), args.weight)  # 返回模型的所有内容
    plt.show()
    return model

def train_modelmulticlasses(model, criterion, optimizer, dataload, lr_scheduler):
    num_epochs=args.num_epochs
    loss_record=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataload:
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            y=y.to(device)
            y=torch.squeeze(y, 1)
            labels = y.long()

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            # print(lr_scheduler.get_lr()[0])
            optimizer.step()
            lr_scheduler.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            loss_record.append(loss.item())
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_data = pd.DataFrame( data=loss_record)
    loss_data.to_csv(args.loss_record)
    plt.plot(loss_data)
    torch.save(model.state_dict(), args.weight)  # 返回模型的所有内容
    plt.show()
    return model
# 训练模型
def train():
    if args.choose_net=="Unet":
        model = my_unet.UNet(3, 1).to(device)
    if args.choose_net=="My_Unet":
        model = my_unet.My_Unet2(3, 1).to(device)
    elif args.choose_net=="Enet":
        model = enet.ENet(num_classes=13).to(device)
    elif args.choose_net=="Segnet":
        model = segnet.SegNet(3,13).to(device)
    elif args.choose_net == "CascadNet":
        model = my_cascadenet.CascadeNet(3, 1).to(device)

    elif args.choose_net == "my_drsnet_A":
        model = my_drsnet.MultiscaleSENetA(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "my_drsnet_B":
        model = my_drsnet.MultiscaleSENetB(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "my_drsnet_C":
        model = my_drsnet.MultiscaleSENetC(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "my_drsnet_A_direct_skip":
        model = my_drsnet.MultiscaleSENetA_direct_skip(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "SEResNet":
        model = my_drsnet.SEResNet18(in_ch=3, out_ch=1).to(device)

    elif args.choose_net == "resnext_unet":
        model = resnext_unet.resnext50(in_ch=3,out_ch=1).to(device)
    elif args.choose_net == "resnet50_unet":
        model = resnet50_unet.UNetWithResnet50Encoder(in_ch=3,out_ch=1).to(device)
    elif args.choose_net == "unet_nest":
        model = unet_nest.UNet_Nested(3,2).to(device)
    elif args.choose_net == "unet_res34":
        model = unet_res34.Resnet_Unet(3,1).to(device)
    elif args.choose_net == "trangle_net":
        model = mytrangle_net.trangle_net(3,1).to(device)
    elif args.choose_net == "dfanet":
        ch_cfg = [[8, 48, 96],
                  [240, 144, 288],
                  [240, 144, 288]]
        model = dfanet.DFANet(ch_cfg,3,1).to(device)
    elif args.choose_net == "lednet":
        model = lednet.Net(num_classes=1).to(device)
    elif args.choose_net == "cgnet":
        model = cgnet.Context_Guided_Network(classes=1).to(device)
    elif args.choose_net == "pspnet":
        model = pspnet.PSPNet(1).to(device)
    elif args.choose_net == "bisenet":
        model = bisenet.BiSeNet(1, 'resnet18').to(device)
    elif args.choose_net == "espnet":
        model = espnet.ESPNet(classes=1).to(device)
    elif args.choose_net == "fddwnet":
        model = fddwnet.Net(classes=1).to(device)
    elif args.choose_net == "contextnet":
        model = contextnet.ContextNet(classes=1).to(device)
    elif args.choose_net == "linknet":
        model = linknet.LinkNet(classes=1).to(device)
    elif args.choose_net == "edanet":
        model = edanet.EDANet(classes=1).to(device)
    elif args.choose_net == "erfnet":
        model = erfnet.ERFNet(classes=1).to(device)

    from collections import OrderedDict

    loadpretrained=0
    # 0:no loadpretrained model
    # 1:loadpretrained model to original network
    # 2:loadpretrained model to new network
    if loadpretrained == 1:
        model.load_state_dict(torch.load(args.weight))

    elif loadpretrained==2:
        model = my_drsnet.MultiscaleSENetA(in_ch=3,out_ch=1).to(device)
        model_dict=model.state_dict()
        pretrained_dict = torch.load(args.weight)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(torch.load(args.weight))
        # pretrained_dict = {k: v for k, v in model.items() if k in model}  # filter out unnecessary keys
        # model.update(pretrained_dict)
        # model.load_state_dict(model)

    # 计算模型参数量和计算量FLOPs
    dsize = (1, 3, 128, 192)
    inputs = torch.randn(dsize).to(device)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    print(" %.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
    batch_size = args.batch_size

    # 加载数据集
    liver_dataset = LiverDataset("data/train_camvid/", transform=x_transform, target_transform=y_transform)
    len_img = liver_dataset.__len__()
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=24)


    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度

    
    # 梯度下降
    # optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # # Observe that all parameters are being optimized

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    # 每n个epoches来一次余弦退火
    cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10*int(len_img/batch_size), eta_min=0.00001)

    multiclass = 1
    if multiclass==1:
        # 损失函数
        class_weights =np.array([0.,6.3005947,4.31063664,34.09234699,50.49834979,3.88280945,
         50.49834979,8.91626081,47.58477105, 29.41289083, 18.95706775, 37.84558871,
         39.3477858])#camvid
        # class_weights = weighing(dataloader, 13, c=1.02)
        class_weights = torch.from_numpy(class_weights).float().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # criterion = LovaszLossSoftmax()
        # criterion = torch.nn.MSELoss()
        train_modelmulticlasses(model, criterion, optimizer, dataloader, cosine_lr_scheduler)
    else:
        # 损失函数
        # criterion = LovaszLossHinge()
        # weights=[0.2]
        # weights=torch.Tensor(weights).to(device)
        # # criterion = torch.nn.CrossEntropyLoss(weight=weights)
        criterion = torch.nn.BCELoss()
        # criterion =focal_loss.FocalLoss(1)
        train_model(model, criterion, optimizer, dataloader, cosine_lr_scheduler)


def test_img(src_path, label_path):
    model_enet = enet.ENet(num_classes=1).to(device)
    model_segnet = segnet.SegNet(3, 1).to(device)
    model_my_mulSE_A = my_drsnet.MultiscaleSENetA(3, 1).to(device)
    model_my_mulSE_B = my_drsnetmy_drsnet.MultiscaleSENetB(3, 1).to(device)
    model_my_mulSE_C = my_drsnet.MultiscaleSENetC(3, 1).to(device)
    model_my_mulSE_A_direct_skip=my_drsnet.MultiscaleSENetA_direct_skip(3, 1).to(device)
    model_SEResNet18 =my_drsnet.SEResNet18(in_ch=3, out_ch=1).to(device)


    ch_cfg = [[8, 48, 96],
              [240, 144, 288],
              [240, 144, 288]]
    model_dfanet = dfanet.DFANet(ch_cfg, 3, 1).to(device)
    model_cgnet = cgnet.Context_Guided_Network(1).to(device)
    model_lednet = lednet.Net(num_classes=1).to(device)
    model_bisenet = bisenet.BiSeNet(1, 'resnet18').to(device)

    model_fddwnet = fddwnet.Net(classes=1).to(device)
    model_contextnet = contextnet.ContextNet(classes=1).to(device)
    model_linknet = linknet.LinkNet(classes=1).to(device)
    model_edanet = edanet.EDANet(classes=1).to(device)
    model_erfnet = erfnet.ERFNet(classes=1).to(device)

    model_enet.load_state_dict(torch.load("./weight/enet_weight.pth"))
    model_enet.eval()
    model_segnet.load_state_dict(torch.load("./weight/segnet_weight.pth"))
    model_segnet.eval()

    model_my_mulSE_A.load_state_dict(torch.load("./weight/my_drsnet_A_weight.pth"))
    model_my_mulSE_A.eval()
    model_my_mulSE_B.load_state_dict(torch.load("./weight/my_drsnet_B_weight.pth"))
    model_my_mulSE_B.eval()
    model_my_mulSE_C.load_state_dict(torch.load("./weight/my_drsnet_C_weight.pth"))
    model_my_mulSE_C.eval()
    model_my_mulSE_A_direct_skip.load_state_dict(torch.load("./weight/my_drsnet_A_direct_skip_weight.pth"))
    model_my_mulSE_A_direct_skip.eval()
    model_SEResNet18.load_state_dict(torch.load("./weight/SEResNet18_weight.pth"))
    model_SEResNet18.eval()



    model_dfanet.load_state_dict(torch.load("./weight/dfanet.pth"))
    model_dfanet.eval()
    model_cgnet.load_state_dict(torch.load("./weight/cgnet.pth"))
    model_cgnet.eval()
    model_lednet.load_state_dict(torch.load("./weight/lednet.pth"))
    model_lednet.eval()
    model_bisenet.load_state_dict(torch.load("./weight/bisenet.pth"))
    model_bisenet.eval()

    model_fddwnet.load_state_dict(torch.load("./weight/fddwnet.pth"))
    model_fddwnet.eval()
    model_contextnet.load_state_dict(torch.load("./weight/contextnet.pth"))
    model_contextnet.eval()
    model_linknet.load_state_dict(torch.load("./weight/linknet.pth"))
    model_linknet.eval()
    model_edanet.load_state_dict(torch.load("./weight/edanet.pth"))
    model_edanet.eval()
    model_erfnet.load_state_dict(torch.load("./weight/erfnet.pth"))
    model_erfnet.eval()

    src = Image.open(src_path)
    src = src.resize((128, 192))
    src = x_transform(src)
    src = src.to(device)
    src = torch.unsqueeze(src, 0)

    y_enet = model_enet(src)
    # label = label.to(device)
    y_enet = y_enet.cpu()
    y_enet = y_enet.detach().numpy().reshape(192, 128)

    y_segnet = model_segnet(src)
    # label = label.to(device)
    y_segnet = y_segnet.cpu()
    y_segnet = y_segnet.detach().numpy().reshape(192, 128)

    y_my_mulSE_A = model_my_mulSE_A(src)
    # label = label.to(device)
    y_my_mulSE_A = y_my_mulSE_A.cpu()
    y_my_mulSE_A = y_my_mulSE_A.detach().numpy().reshape(192, 128)

    y_my_mulSE_B = model_my_mulSE_B(src)
    # label = label.to(device)
    y_my_mulSE_B = y_my_mulSE_B.cpu()
    y_my_mulSE_B = y_my_mulSE_B.detach().numpy().reshape(192, 128)

    y_my_mulSE_C = model_my_mulSE_C(src)
    # label = label.to(device)
    y_my_mulSE_C = y_my_mulSE_C.cpu()
    y_my_mulSE_C = y_my_mulSE_C.detach().numpy().reshape(192, 128)

    y_my_mulSE_A_direct_skip = model_my_mulSE_A_direct_skip(src)
    # label = label.to(device)
    y_my_mulSE_A_direct_skip = y_my_mulSE_A_direct_skip.cpu()
    y_my_mulSE_A_direct_skip = y_my_mulSE_A_direct_skip.detach().numpy().reshape(192, 128)

    y_SEResNet18 = model_SEResNet18(src)
    # label = label.to(device)
    y_SEResNet18 = y_SEResNet18.cpu()
    y_SEResNet18 = y_SEResNet18.detach().numpy().reshape(192, 128)





    y_dfanet = model_dfanet(src)
    # label = label.to(device)
    y_dfanet = y_dfanet.cpu()
    y_dfanet = y_dfanet.detach().numpy().reshape(192, 128)

    y_cgnet = model_cgnet(src)
    # label = label.to(device)
    y_cgnet = y_cgnet.cpu()
    y_cgnet = y_cgnet.detach().numpy().reshape(192, 128)

    y_lednet = model_lednet(src)
    # label = label.to(device)
    y_lednet = y_lednet.cpu()
    y_lednet = y_lednet.detach().numpy().reshape(192, 128)

    y_bisenet = model_bisenet(src)
    # label = label.to(device)
    y_bisenet = y_bisenet.cpu()
    y_bisenet = y_bisenet.detach().numpy().reshape(192, 128)

    y_fddwnet = model_fddwnet(src)
    # label = label.to(device)
    y_fddwnet = y_fddwnet.cpu()
    y_fddwnet = y_fddwnet.detach().numpy().reshape(192, 128)

    y_contextnet = model_contextnet(src)
    # label = label.to(device)
    y_contextnet = y_contextnet.cpu()
    y_contextnet = y_contextnet.detach().numpy().reshape(192, 128)

    y_linknet = model_linknet(src)
    # label = label.to(device)
    y_linknet = y_linknet.cpu()
    y_linknet = y_linknet.detach().numpy().reshape(192, 128)

    y_edanet = model_edanet(src)
    # label = label.to(device)
    y_edanet = y_edanet.cpu()
    y_edanet = y_edanet.detach().numpy().reshape(192, 128)

    y_erfnet = model_erfnet(src)
    # label = label.to(device)
    y_erfnet = y_erfnet.cpu()
    y_erfnet = y_erfnet.detach().numpy().reshape(192, 128)

    y_enet = (y_enet > 0.5).astype(int) * 255
    y_segnet = (y_segnet > 0.5).astype(int) * 255
    y_my_mulSE_A = (y_my_mulSE_A > 0.5).astype(int) * 255
    y_my_mulSE_B = (y_my_mulSE_B > 0.5).astype(int) * 255
    y_my_mulSE_C = (y_my_mulSE_C > 0.5).astype(int) * 255
    y_my_mulSE_A_direct_skip = (y_my_mulSE_A_direct_skip > 0.5).astype(int) * 255
    y_SEResNet18 = (y_SEResNet18 > 0.5).astype(int) * 255



    y_dfanet = (y_dfanet > 0.5).astype(int) * 255
    y_cgnet = (y_cgnet > 0.5).astype(int) * 255
    y_lednet = (y_lednet > 0.5).astype(int) * 255
    y_bisenet = (y_bisenet > 0.5).astype(int) * 255

    y_fddwnet = (y_fddwnet > 0.5).astype(int) * 255
    y_contextnet = (y_contextnet > 0.5).astype(int) * 255
    y_linknet = (y_linknet > 0.5).astype(int) * 255
    y_edanet = (y_edanet > 0.5).astype(int) * 255
    y_erfnet = (y_erfnet > 0.5).astype(int) * 255

    src1 = Image.open(src_path)
    src1 = src1.resize((128, 192))
    label = Image.open(label_path)
    label = label.resize((128, 192))
    label = np.array(label) * 255
    src1.save("./data/result/" + "_src.png")
    cv2.imwrite("./data/result/" + "_label.png", label)
    cv2.imwrite("./data/result/" + "enet_predict.png", y_enet)
    cv2.imwrite("./data/result/" + "segnet_predict.png", y_segnet)
    cv2.imwrite("./data/result/" + "my_drsnet_A_predict.png", y_my_mulSE_A)
    cv2.imwrite("./data/result/" + "my_drsnet_B_predict.png", y_my_mulSE_B)
    cv2.imwrite("./data/result/" + "my_drsnet_C_predict.png", y_my_mulSE_C)
    cv2.imwrite("./data/result/" + "my_drsnet_A_direct_skip_predict.png", y_my_mulSE_A_direct_skip)
    cv2.imwrite("./data/result/" + "y_SEResNet18_predict.png", y_SEResNet18)

    cv2.imwrite("./data/result/" + "dfanet_predict.png", y_dfanet)
    cv2.imwrite("./data/result/" + "cgnet_predict.png", y_cgnet)
    cv2.imwrite("./data/result/" + "lednet_predict.png", y_lednet)
    cv2.imwrite("./data/result/" + "bisenet_predict.png", y_bisenet)

    cv2.imwrite("./data/result/" + "fddwnet_predict.png", y_fddwnet)
    cv2.imwrite("./data/result/" + "contextnet_predict.png", y_contextnet)
    cv2.imwrite("./data/result/" + "linknet_predict.png", y_linknet)
    cv2.imwrite("./data/result/" + "edanet_predict.png", y_edanet)
    cv2.imwrite("./data/result/" + "erfnet_predict.png", y_erfnet)

    return 0



# 测试
def test():
    if args.choose_net=="Unet":
        model = my_unet.UNet(3, 1).to(device)
    if args.choose_net=="My_Unet":
        model = my_unet.My_Unet2(3, 1).to(device)
    elif args.choose_net=="Enet":
        model = enet.ENet(num_classes=13).to(device)
    elif args.choose_net=="Segnet":
        model = segnet.SegNet(3,1).to(device)
    elif args.choose_net == "CascadNet":
        model = my_cascadenet.CascadeNet(3, 1).to(device)

    elif args.choose_net == "my_drsnet_A":
        model = my_drsnet.MultiscaleSENetA(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "my_drsnet_B":
        model = my_drsnet.MultiscaleSENetB(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "my_drsnet_C":
        model = my_drsnet.MultiscaleSENetC(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "my_drsnet_A_direct_skip":
        model = my_drsnet.MultiscaleSENetA_direct_skip(in_ch=3, out_ch=1).to(device)
    elif args.choose_net == "SEResNet":
        model = my_drsnet.SEResNet18(in_ch=3, out_ch=1).to(device)

    elif args.choose_net == "resnext_unet":
        model = resnext_unet.resnext50(in_ch=3,out_ch=1).to(device)
    elif args.choose_net == "resnet50_unet":
        model = resnet50_unet.UNetWithResnet50Encoder(in_ch=3,out_ch=1).to(device)
    elif args.choose_net == "unet_res34":
        model = unet_res34.Resnet_Unet(in_ch=3,out_ch=1).to(device)
    elif args.choose_net == "dfanet":
        ch_cfg = [[8, 48, 96],
                  [240, 144, 288],
                  [240, 144, 288]]
        model = dfanet.DFANet(ch_cfg,3,1).to(device)
    elif args.choose_net == "cgnet":
        model = cgnet.Context_Guided_Network(1).to(device)
    elif args.choose_net == "lednet":
        model = lednet.Net(num_classes=1).to(device)
    elif args.choose_net == "bisenet":
        model = bisenet.BiSeNet(1, 'resnet18').to(device)
    elif args.choose_net == "espnet":
        model = espnet.ESPNet(classes=1).to(device)
    elif args.choose_net == "pspnet":
        model = pspnet.PSPNet(1).to(device)
    elif args.choose_net == "fddwnet":
        model = fddwnet.Net(classes=1).to(device)
    elif args.choose_net == "contextnet":
        model = contextnet.ContextNet(classes=1).to(device)
    elif args.choose_net == "linknet":
        model = linknet.LinkNet(classes=1).to(device)
    elif args.choose_net == "edanet":
        model = edanet.EDANet(classes=1).to(device)
    elif args.choose_net == "erfnet":
        model = erfnet.ERFNet(classes=1).to(device)
    dsize = (1, 3, 128, 192)
    inputs = torch.randn(dsize).to(device)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    print(" %.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))

    model.load_state_dict(torch.load(args.weight))
    liver_dataset = LiverDataset("data/val_camvid", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)  # batch_size默认为1
    model.eval()

    metric = SegmentationMetric(13)
    # import matplotlib.pyplot as plt
    # plt.ion()
    multiclass=1
    mean_acc,mean_miou=[],[]

    alltime=0.0
    with torch.no_grad():
        for x, y_label in dataloaders:
            x=x.to(device)
            start = time.time()
            y = model(x)
            usingtime = time.time() - start
            alltime=alltime+usingtime

            if multiclass==1:
                # predict输出处理：
                # https://www.cnblogs.com/ljwgis/p/12313047.html
                y = F.sigmoid(y)
                y = y.cpu()
                # y = torch.squeeze(y).numpy()
                y = torch.argmax(y.squeeze(0),dim=0).data.numpy()
                print(y.max(),y.min())
                # y_label = y_label[0]
                y_label =torch.squeeze(y_label).numpy()
            else:
                y = y.cpu()
                y = torch.squeeze(y).numpy()
                y_label = torch.squeeze(y_label).numpy()


                # img_y = y*127.5


                if args.choose_net == "Unet":
                    y = (y>0.5)
                elif args.choose_net == "My_Unet":
                    y = (y>0.5)
                elif args.choose_net == "Enet":
                    y = (y>0.5)
                elif args.choose_net == "Segnet":
                    y = (y > 0.5)
                elif args.choose_net == "Scnn":
                    y = (y > 0.5)
                elif args.choose_net == "CascadNet":
                    y = (y > 0.8)

                elif args.choose_net == "my_drsnet_A":
                    y = (y > 0.5)
                elif args.choose_net == "my_drsnet_B":
                    y = (y > 0.5)
                elif args.choose_net == "my_drsnet_C":
                    y = (y > 0.5)
                elif args.choose_net == "my_drsnet_A_direct_skip":
                    y = (y > 0.5)
                elif args.choose_net == "SEResNet":
                    y = (y > 0.5)


                elif args.choose_net == "resnext_unet":
                    y = (y > 0.5)
                elif args.choose_net == "resnet50_unet":
                    y = (y > 0.5)
                elif args.choose_net == "unet_res34":
                    y = (y > 0.5)
                elif args.choose_net == "dfanet":
                    y = (y > 0.5)
                elif args.choose_net == "cgnet":
                    y = (y > 0.5)
                elif args.choose_net == "lednet":
                    y = (y > 0.5)
                elif args.choose_net == "bisenet":
                    y = (y > 0.5)
                elif args.choose_net == "pspnet":
                    y = (y > 0.5)
                elif args.choose_net == "fddwnet":
                    y = (y > 0.5)
                elif args.choose_net == "contextnet":
                    y = (y > 0.5)
                elif args.choose_net == "linknet":
                    y = (y > 0.5)
                elif args.choose_net == "edanet":
                    y = (y > 0.5)
                elif args.choose_net == "erfnet":
                    y = (y > 0.5)

            img_y = y.astype(int).squeeze()
            print(y_label.shape, img_y.shape)
            image = np.concatenate((img_y,y_label))


            y_label=y_label.astype(int)
            metric.addBatch(img_y, y_label)
            acc = metric.classPixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            # confusionMatrix=metric.genConfusionMatrix(img_y, y_label)
            mean_acc.append(acc[1])
            mean_miou.append(mIoU)
            # print(acc, mIoU,confusionMatrix)
            print(acc, mIoU)
            plt.imshow(image*5)
            plt.pause(0.1)
            plt.show()
    # 计算时需封印acc和miou计算部分

    print("Took ",alltime , "seconds")
    print("Took",alltime/638.0, "s/perimage")
    print("FPS", 1/(alltime / 638.0))
    print("average acc:%0.6f  average miou:%0.6f" % (np.mean(mean_acc), np.mean(mean_miou)))

def train_dehaze_model(modeldehaze, modelsegmentation,criteriondehaze, criterionsegmentation,optimizerdehaze,optimizersegmentation,
                       dataloaddehaze,num_epochs=6):
    loss_record=[[],[]]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataloaddehaze.dataset)
        epoch_dehaze_loss = 0
        epoch_dsegmentation_loss = 0
        step = 0  # minibatch数
        for src,rain,mask in  dataloaddehaze:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizerdehaze.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            optimizersegmentation.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            src_torch = src.to(device)
            rain_torch = rain.to(device)
            mask_torch = mask.to(device)

            dehaze_output = modeldehaze(rain_torch)  # 前向传播
            dehaze_loss = criteriondehaze(dehaze_output, src_torch)  # 计算损失
            dehaze_loss.backward(retain_graph=True)  # 梯度下降,计算出梯度
            optimizerdehaze.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新


            segmentation_output = modelsegmentation(dehaze_output)  # 前向传播
            segmentation_loss = criterionsegmentation(segmentation_output, mask_torch)  # 计算损失
            segmentation_loss.backward()  # 梯度下降,计算出梯度
            optimizersegmentation.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新

            epoch_dehaze_loss += dehaze_loss.item()
            epoch_dsegmentation_loss += segmentation_loss.item()
            loss_record[0].append(dehaze_loss.item())
            loss_record[1].append(segmentation_loss.item())
            step += 1
            print("%d/%d,dehaze_loss:%0.3f,loss_segmentation:%0.3f" % (step, dataset_size // dataloaddehaze.batch_size,
                                                                       dehaze_loss.item(), segmentation_loss.item()))
        print("epoch %d epoch_dehaze_loss:%0.3f  epoch_dsegmentation_loss:%0.3f" % (epoch, epoch_dehaze_loss,epoch_dsegmentation_loss))
    torch.save(modeldehaze.state_dict(), "dehaze.pth")  # 返回模型的所有内容
    torch.save(modelsegmentation.state_dict(), args.weight)  # 返回模型的所有内容c

    loss_data = pd.DataFrame(data=loss_record)
    loss_data.to_csv(args.loss_record)
    plt.plot(loss_data)
    plt.show()

def trainwithdehaze():
    model_dehaze=aodnet.AODnet().to(device)
    dsize = (3, 1, 256, 256)
    # inputs1 = torch.randn(dsize).to(device)
    # total_ops, total_params = profile(model_dehaze, (inputs1,), verbose=False)
    # print(" %.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
    if args.choose_net=="Unet":
        model_segmentation = my_unet.UNet(3, 1).to(device)
    elif args.choose_net=="Enet":
        model_segmentation = enet.ENet(num_classes=1).to(device)
    elif args.choose_net=="Segnet":
        model_segmentation = segnet.SegNet(3,1).to(device)

    # inputs2 = torch.randn(dsize).to(device)
    # total_ops, total_params = profile(model_segmentation, (inputs2,), verbose=False)
    # print(" %.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
    batch_size = args.batch_size
    # dehaze的损失函数
    criterion_dehaze = torch.nn.MSELoss()
    # dehaze的优化函数
    optimizer_dehaze = optim.Adam(model_dehaze.parameters())  # model.parameters():Returns an iterator over module parameters

    # 语义分割的损失函数
    criterion_segmentation = torch.nn.BCELoss()
    # 语义分割的优化函数
    optimizer_segmentation = optim.Adam(model_segmentation.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    dataset_dehaze = LiverDataset_three("data/train_dehaze/", transform=x_transform, target_transform=y_transform)
    dataloader_dehaze = DataLoader(dataset_dehaze, batch_size=batch_size, shuffle=True, num_workers=4)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_dehaze_model(model_dehaze, model_segmentation,criterion_dehaze, criterion_segmentation,
                       optimizer_dehaze, optimizer_segmentation,dataloader_dehaze,num_epochs=6)

# 测试
def testwithdehaze():
    model_dehaze = aodnet.AODnet()
    model_dehaze.load_state_dict(torch.load("./weight/dehaze.pth", map_location='cpu'))

    if args.choose_net=="Unet":
        model_segmentation = my_unet.UNet(3, 1)
    elif args.choose_net=="Enet":
        model_segmentation = enet.ENet(num_classes=1)
    elif args.choose_net=="Segnet":
        model_segmentation = segnet.SegNet(3,1)
    model_segmentation.load_state_dict(torch.load(args.weight, map_location='cpu'))
    liver_dataset = LiverDataset_three("data/val_dehaze", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)  # batch_size默认为1
    model_dehaze.eval()
    model_segmentation.eval()

    metric = SegmentationMetric(2)

    # import matplotlib.pyplot as plt
    # plt.ion()
    mean_acc,mean_miou=[],[]
    with torch.no_grad():
        for src,rain,mask in dataloaders:
            y = model_dehaze(src)
            y1=model_segmentation(y)
            y1=torch.squeeze(y1).numpy()
            y_label=torch.squeeze(mask).numpy()
            y_label = y_label * 255
            y1 = y1 * 127.5
            # print(y_label.shape,y.shape)
            image = np.concatenate((y_label, y1))

            if args.choose_net == "Unet":
                img_y = (y1>0.5)
            elif args.choose_net == "Enet":
                img_y = (y1>0.5)
            elif args.choose_net == "Segnet":
                img_y = (y1 > 0.5)
            elif args.choose_net == "Scnn":
                img_y = (y1 > 0.5)
            img_y=img_y.astype(int)

            y_label=y_label.astype(int)
            metric.addBatch(img_y, y_label)
            acc = metric.pixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            # confusionMatrix=metric.genConfusionMatrix(img_y, y_label)
            mean_acc.append(acc)
            mean_miou.append(mIoU)
            # print(acc, mIoU,confusionMatrix)
            print(acc, mIoU)
            # plt.imshow(image)
            # plt.pause(0.01)
            # plt.show()
    print("average acc:%0.6f  average miou:%0.6f" % (np.mean(mean_acc),np.mean(mean_miou)))


if __name__ == '__main__':
    # 参数解析
    # Segnet:29.44 | 40.10
    # without rain:

    # Unet:26.50 | 53.37
    # with rain:average acc:0.973897  average miou:0.946637
    #           average acc:0.973128  average miou:0.944792
    # average acc:0.967947  average miou:0.935041
    # average acc:0.972805  average miou:0.944366
    # without rain:

    # Unet2(resunet): 37.32 | 74.09
    # with rain: average acc:0.974125  average miou:0.947126

    # resnet-unet: 147.81 | 53.73
    # withrain:average acc:0.967668  average miou:0.934363

    # Enet:0.35 | 0.51
    # with rain:average acc:0.960504  average miou:0.920452
    # average acc:0.957815  average miou:0.915256
    # without rain:0.9751303473824252  0.9491616651984517

    # mltiscaleSE: 8.72 | 26.06
    # with rain:
    # 1*3,3*1 in_ch=36:               average acc:0.968561  average miou:0.935664
    # 1*3,3*1 in_ch=48:               average acc:0.967267  average miou:0.933177
    # 1*5, in_ch=36:                  average acc:0.966435  average miou:0.931600
    # without SE,in_ch=36:            average acc:0.964121  average miou:0.926902
    # without c1_concat += x,in_ch=36:average acc:0.963467  average miou:0.926219
    # MultiscaleInception2SE,in_ch=36:average acc:0.942046  average miou:0.882606
    # 1*3,3*1 in_ch=36,1*3 dilated:   average acc:0.966872  average miou:0.932287
    # 16*16:average acc:0.952947  average miou:0.905644
    # res+36:average acc:0.969880  average miou:0.938617
    # res+48:average acc:0.969971  average miou:0.938996
    # 9.00 | 21.42 dilation+res+36:average acc:0.973287  average miou:0.945590
    #  8.64 | 16.69 dilation+res+36+singleconv:average acc:0.971705  average miou:0.942272

    # dilation+res+48:average acc:0.972909  average miou:0.944813
    #  9.02 | 23.55  dilation+res+36+up:average acc:0.970213  average miou:0.939304
    # 5.28 | 11.51  dilation+res+24:average acc:0.967548  average miou:0.934102
    # dilation+rMultiscaleSE+36: average acc:0.971284  average miou:0.941487
    #  8.80 | 21.43 dilation+rMultiscaleSE+36:average acc:0.971345  average miou:0.941526




    # Icome
    # withoutrain
    # mltiscaleSE:average acc:0.949276  average miou:0.869269
    # unet:average acc:0.959325  average miou:0.895530
    # Enet:average acc:0.803466  average miou:0.806792
    # Segnet：average acc:0.946971  average miou:0.865122

    # mltiscaleSENew: average acc:0.944549  average miou:0.857436

    # average acc:0.910289  average miou:0.777546
    # average acc:0.940777  average miou:0.850844
    # average acc:0.943011  average miou:0.856429
    # average acc:0.942675  average miou:0.856389
    # average acc:0.939823  average miou:0.847479
    # average acc:0.953173  average miou:0.880033

    # resnet34-unet:average acc:0.936603  average miou:0.841389
    # without rain:
    # Unet:average acc:0.891490  average miou:0.875689
    # average acc:0.919044  average miou:0.905937
    # mltiscaleSENew:average acc:0.907125  average miou:0.885679
    # average acc:0.904860  average miou:0.889577



    # Icome
    # BiSenET: 12.40 | 2.02:average acc:0.897412  average miou:0.895928  Took 0.0044446686592221635  s/perimage         FPS 224.9886496994816
    # Enet:average acc:0.803466  average miou:0.806792                           Took 0.05664537077056431 s/perimage
    # dfanet: 2.09 | 0.27:average acc:0.695921  average miou:0.634010            Took 0.08609732110261031 s/perimage
    # LEDNET: 0.92 | 1.43:大小图像尺度均不收敛
    # CGNet: 0.49 | 0.86:average acc:0.800661  average miou:0.768480             Took 0.055695938798131554 s/perimage
    # MultiscaleNet: 0.54 | 0.52:                                                Took 0.030065771815502067 s/perimage
    # 10epoches average acc:0.799616  average miou:0.781600  initial learningrate:0.008
    # 20epoches average acc:0.845499  average miou:0.827916  initial learningrate:0.002
    # pspnet:65.57 | 24.94：average acc:0.834110  average miou:0.821406
    # fddwnet:0.81 | 0.62:average acc:0.805103  average miou:0.792511 Took 0.08420729193988787 s/perimage FPS 14.082789965440309



    # UAS withrain:
    # BiSenET: 12.40 | 2.02:average acc:0.915040  average miou:0.906446  Took 0.0431481457997432 s/perimage
    # segnet:29.44 | 15.04：average acc:0.958093  average miou:0.943488
    #  MultiscaleSEnEW3:
    # 2 2 2:0.55 | 0.20:average acc:0.938589  average miou:0.916183
    # 2 2 2:0.54 | 0.20:average acc:0.944903  average miou:0.921398     Took 0.021915494087721487 s/perimage
    # 3 2 2:0.53 | 0.20:average acc:0.932374  average miou:0.902905
    # 2 2 3:0.37 | 0.20:average acc:0.920155  average miou:0.907908
    # dfanet:2.09 | 0.10:average acc:0.886232  average miou:0.784329     Took 0.08054251282192697 s/perimage
    # LEDNET:  0.92 | 0.53:0.961439  average miou:0.842492               Took 0.049218488711174756 s/perimage
    # CGNet:0.49 | 0.32:average acc:0.936055  average miou:0.901427      Took 0.04885384094752488 s/perimage
    # Enet:0.35 | 0.19:average acc:0.935536  average miou:0.917971       Took 0.048936753811133694 s/perimage
    # pspnet：65.57 | 24.94：average acc:0.936101  average miou:0.919340
    # fddwnet:0.81 | 0.62:average acc:0.961086  average miou:0.896694


    # average acc:0.836187  average miou:0.820888

    # camvid:
    # ENet:average acc:0.979088  average miou:0.273855 Took 0.011355369950758924 s/perimage FPS 104.43284640573941

    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument('--action', type=str,help='train or test or test_img or trainwithdehaze or testwithdehaze', default="train")  # 添加参数
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.008)  # 添加参数
    parser.add_argument('--choose_net', type=str,
                        help='erfnet or edanet or linknet or contextnet or fddwnet or espnet or bisenet or pspnet or cgnet or lednet or dfanet or trangle_net or'
                             ' unet_nest or resnet50_unet or my_drsnet_A or my_drsnet_B or my_drsnet_C or my_drsnet_A_direct_skip or SEResNet or resnext_unet or Unet or My_Unet or Enet or Segnet or CascadNet or unet_res34',
                        default="Enet")#can not work with class=1: espnet
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file', default="./weight/enet_weight.pth")
    parser.add_argument('--loss_record', type=str, help='the name of theloss_record file', default="enet_loss.csv")
    args = parser.parse_args()
    if args.choose_net == "Unet":
        args.weight='./weight/unet_weight.pth'
        args.loss_record="unet_loss.csv"
    elif args.choose_net == "My_Unet":
        args.weight = './weight/My_Unet.pth'
        args.loss_record = "My_Unet.csv"
    elif args.choose_net == "Enet":
        args.weight='./weight/enet_weight.pth'
        args.loss_record = "enet_loss.csv"
    elif args.choose_net == "Segnet":
        args.weight = './weight/segnet_weight.pth'
        args.loss_record = "segnet_loss.csv"
    elif args.choose_net == "CascadNet":
        args.weight = './weight/cascadenet_weight.pth'
        args.loss_record = "cascadenet_loss.csv"

    elif args.choose_net == "my_drsnet_A":
        args.weight = './weight/my_drsnet_A_weight.pth'
        args.loss_record = "my_drsnet_A_loss.csv"
    elif args.choose_net == "my_drsnet_B":
        args.weight = './weight/my_drsnet_B_weight.pth'
        args.loss_record = "my_drsnet_B_loss.csv"
    elif args.choose_net == "my_drsnet_C":
        args.weight = './weight/my_drsnet_C_weight.pth'
        args.loss_record = "my_drsnet_C_loss.csv"
    elif args.choose_net == "my_drsnet_A_direct_skip":
        args.weight = './weight/my_drsnet_A_direct_skip_weight.pth'
        args.loss_record = "my_drsnet_A_direct_skip_loss.csv"
    elif args.choose_net == "SEResNet":
        args.weight = './weight/SEResNet_weight.pth'
        args.loss_record = "SEResNet_loss.csv"



    elif args.choose_net == "resnext_unet":
        args.weight = './weight/resnext_weight.pth'
        args.loss_record = "resnext_loss.csv"
    elif args.choose_net == "resnet50_unet":
        args.weight = './weight/resnet_unet_weight.pth'
        args.loss_record = "resnet_unet_loss.csv"
    elif args.choose_net == "unet_nest":
        args.weight = './weight/unet_nest_weight.pth'
        args.loss_record = "unet_nest_loss.csv"
    elif args.choose_net == "unet_res34":
        args.weight = './weight/my_unet_res34.pth'
        args.loss_record = "my_unet_res34.csv"
    elif args.choose_net == "trangle_net":
        args.weight = './weight/my_trangle_net.pth'
        args.loss_record = "my_trangle_net.csv"
    elif args.choose_net == "dfanet":
        args.weight = './weight/dfanet.pth'
        args.loss_record = "dfanet.csv"
    elif args.choose_net == "lednet":
        args.weight = './weight/lednet.pth'
        args.loss_record = "lednet.csv"
    elif args.choose_net == "cgnet":
        args.weight = './weight/cgnet.pth'
        args.loss_record = "cgnet.csv"
    elif args.choose_net == "pspnet":
        args.weight = './weight/pspnet.pth'
        args.loss_record = "pspnet.csv"
    elif args.choose_net == "bisenet":
        args.weight = './weight/bisenet.pth'
        args.loss_record = "bisenet.csv"
    elif args.choose_net == "espnet":
        args.weight = './weight/espnet.pth'
        args.loss_record = "espnet.csv"
    elif args.choose_net == "fddwnet":
        args.weight = './weight/fddwnet.pth'
        args.loss_record = "fddwnet.csv"
    elif args.choose_net == "contextnet":
        args.weight = './weight/contextnet.pth'
        args.loss_record = "contextnet.csv"
    elif args.choose_net == "linknet":
        args.weight = './weight/linknet.pth'
        args.loss_record = "linknet.csv"
    elif args.choose_net == "edanet":
        args.weight = './weight/edanet.pth'
        args.loss_record = "edanet.csv"
    elif args.choose_net == "erfnet":
        args.weight = './weight/erfnet.pth'
        args.loss_record = "erfnet.csv"



    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
    elif args.action == 'test_img':
        src_path = "./data/val_withrain/2.jpg"
        label_path = "./data/val_withrain/2.png"
        test_img(src_path,label_path)
    elif args.action == 'trainwithdehaze':
        trainwithdehaze()
    elif args.action == 'testwithdehaze':
        testwithdehaze()