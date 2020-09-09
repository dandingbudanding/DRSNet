import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# 数据矩阵转图片的函数
def MatrixToImage(data):
    data = data/2+0.5
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

# 添加路径，metal文件夹下存放mental类的特征的多个.mat文件
folder = "./spatial_envelope_256x256_static_8outdoorcategories"
path = os.listdir(folder)
#print(os.listdir(r'/Users/hjy/Desktop/blues'))

for each_mat in path:
    if each_mat == '.DS_Store':
        pass
    else:
        first_name, second_name = os.path.splitext(each_mat)
        # 拆分.mat文件的前后缀名字，注意是**路径**
        each_mat = os.path.join(folder, each_mat)
        # print(each_mat)
        array_struct = scio.loadmat(each_mat)
        array_data = array_struct['S']# 取出需要的数字矩阵部分
        new_im = MatrixToImage(array_data)# 调用函数
        plt.imshow(array_data, cmap=plt.cm.gray, interpolation='nearest')
        new_im.show()
        print(first_name)
        new_im.save(first_name+'.bmp')# 保存图片
