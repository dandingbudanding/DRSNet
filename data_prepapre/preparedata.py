# rename the images of UAS dataset
import os
from random import shuffle
src_path="../UAS Dataset/UAS(UESTC All-day Scenery)/src"
label_path="../UAS Dataset/UAS(UESTC All-day Scenery)/label"


class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self,pathsrc,pathlabel):
        self.pathsrc = pathsrc  #表示需要命名处理的文件夹
        self.pathlabel = pathlabel  # 表示需要命名处理的文件夹

    def rename(self):
        filelist = os.listdir(self.pathsrc) #获取文件路径
        shuffle(filelist)
        total_num = len(filelist) #获取文件长度（个数）
        i = 1  #表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.jpg'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                img_src = os.path.join(os.path.abspath(self.pathsrc), item)
                img_src_dst = os.path.join(os.path.abspath(self.pathsrc),str(i) + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                label_item=item.replace("Sight","Label")
                label_item = label_item.replace("jpg", "png")
                img_label = os.path.join(os.path.abspath(self.pathlabel), label_item)
                img_label_dst = os.path.join(os.path.abspath(self.pathlabel),str(i) + '.png')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                #dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(img_src, img_src_dst)
                    os.rename(img_label, img_label_dst)
                    # print ('converting %s to %s ...' % (img_src, img_src_dst))
                    print(i)
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename(src_path,label_path)
    demo.rename()


