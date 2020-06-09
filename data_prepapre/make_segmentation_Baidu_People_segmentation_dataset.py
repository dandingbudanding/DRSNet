import cv2
import numpy as np
import os

import cv2
import numpy as np


def get_noise(img, value=10):
    '''
    #生成噪声图像
    输入： img图像
           value= 大小控制雨滴的多少
    返回图像大小的模糊噪声图像
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # 可以输出噪声看看
    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise


def rain_blur(noise, length=6, angle=0, w=1):
    '''
    将噪声加上运动模糊,模仿雨滴

    输入
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小

    输出带模糊的噪声

    '''

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''

    return blurred


def alpha_rain(rain, img, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果

    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()  # 拷贝一个掩膜
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）

    # cv2.imshow('rain_effct_result', rain_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return rain_result



def add_rain(rain, img, alpha=0.9):
    # 输入雨滴噪声和图像
    # alpha：原图比例因子
    # 显示下雨效果

    # chage rain into  3-dimenis
    # 将二维rain噪声扩张为与原图相同的三通道图像
    rain = np.expand_dims(rain, 2)
    rain = np.repeat(rain, 3, 2)

    # 加权合成新图
    result = cv2.addWeighted(img, alpha, rain, 1 - alpha, 1)
    # cv2.imshow('rain_effct', result)
    # cv2.waitKey()
    # cv2.destroyWindow('rain_effct')


if __name__ == '__main__':
    train_path= "../data/train_Icome"
    test_path = "../data/val_Icome"

    src_path="../../Icomedataset/icome_task2_data/clean_images/images"
    label_path="../../Icomedataset/icome_task2_data/clean_images/profiles"

    src_all=os.listdir(src_path)
    lenof_src=len(src_all)
    test_num=int(lenof_src*0.1)  #分多少给测试集
    for _,file in enumerate(src_all):
        img=cv2.imread(os.path.join(src_path,file))
        label=cv2.imread(os.path.join(label_path,file[:-4]+"-profile.jpg"),0) #0:origin 1:bgr
        label=(~label)//150
        noise = get_noise(img, value=800)
        rain = rain_blur(noise, length=50, angle=-30, w=3)
        img_addrain=alpha_rain(rain, img, beta=0.8)  # 方法一，透明度赋值
        if _<test_num:
            cv2.imwrite(os.path.join(test_path,str(_)+".jpg"),img_addrain)
            cv2.imwrite(os.path.join(test_path, str(_)+ ".png"), label)
        else:
            cv2.imwrite(os.path.join(train_path, str(_-test_num) + ".jpg"), img_addrain)
            cv2.imwrite(os.path.join(train_path, str(_-test_num) + ".png"), label)
        print(_)
