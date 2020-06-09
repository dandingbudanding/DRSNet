import torch.utils.data as data
import os
import PIL.Image as Image
import cv2
import numpy as np
import torch


# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        n = len(os.listdir(root)) // 2  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        print(n)

        imgs = []
        for i in range(n):
            img = os.path.join(root, "%d.jpg" % i)  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            mask = os.path.join(root, "%d.png" % i)
            imgs.append([img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        # img_x = cv2.imread(x_path,-1)
        # img_x=cv2.resize(img_x,(128,192),interpolation=cv2.INTER_NEAREST)
        # # img_x = img_x.resize((256, 256))
        # img_y = cv2.imread(y_path,-1)
        # img_y=cv2.resize(img_y,(128,192),interpolation=cv2.INTER_NEAREST)
        # # img_y = img_y.resize((256, 256))

        # # a,b,c,d=img_x.max(),img_x.min(),img_y.max(), img_y.min()
        # # print(a, b, c, d)
        # img_x = img_x / 255.0
        # img_x = img_x.astype(np.float32)
        # img_x = torch.from_numpy(img_x)
        # img_y = torch.LongTensor(img_y)
        # 512,768
        img_x = Image.open(x_path)
        img_x = img_x.resize((256,384))
        # img_x = img_x.resize((256, 256))
        img_y = Image.open(y_path)
        img_y = img_y.resize((256,384))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        img_y=img_y*255.0
        # img_y = img_y.long()
        img_y = torch.squeeze(img_y)
        return img_x, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]

class LiverDataset_three(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        n = len(os.listdir(root)) // 3  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        print(n)

        imgs = []
        for i in range(n):
            img = os.path.join(root, "%d.jpg" % i)  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            rain_img = os.path.join(root, "%d.bmp" % i)
            mask = os.path.join(root, "%d.png" % i)

            imgs.append([img, rain_img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, rainimg_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_x=img_x.resize((256,256))
        img_rain = Image.open(rainimg_path)
        img_rain = img_rain.resize((256, 256))
        img_y = Image.open(y_path)
        img_y=img_y.resize((256, 256))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.transform is not None:
            img_rain = self.transform(img_rain)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_rain, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]


def weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()
        flat_label=flat_label.astype(int)

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))
    class_weights[0]=0

    print("Class weights:", class_weights)

    return class_weights