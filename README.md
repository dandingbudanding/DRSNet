# DRSNet
a real-time lightweight network for semantic segmentation in rainy environments, shows high performance compared with other lightweight semantic segmentation networks(BiSeNet、CGNet、ContextNet、DFANet、EDANet、ENET、ERFNet、ESPNet、FDDWNet、LEDNet、LinkNet、UNet（add none-local、use heavy backbones（resnet、resnext、SEresnet）、change the structures)、PSPNet、SEGNet、UNet_nest.

add lovaze-softmax loss\focal loss for segmentation

environments:
win10 or Ubuntu18.04
torch 1.1.0  CUDA10.1

for train：
sudo python3 --action="train" --choose_net="NET"
NET: is the parameter you can choose in main.py file, you also can change the parameters directly in main.py file.

for test:
sudo python3 --action="test" --choose_net="NET"

to change datasets:
just change the fold name of datasets in main.py file

any problems, youcan contact with me, my email is 2463908977@qq.com.
