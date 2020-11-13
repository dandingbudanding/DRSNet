# DRSNet
a real-time lightweight network for semantic segmentation in rainy environments, shows high performance compared with other lightweight semantic segmentation networks(BiSeNet、CGNet、ContextNet、DFANet、EDANet、ENET、ERFNet、ESPNet、FDDWNet、LEDNet、LinkNet、UNet（add none-local、use heavy backbones（resnet、resnext、SEresnet）、change the structures)、PSPNet、SEGNet、UNet_nest.

add lovaze-softmax loss\focal loss for segmentation

environments:
Ubuntu18.04
torch 1.1.0  CUDA10.1

For train：
sudo python3 --action="train" --choose_net="NET"
NET: is the parameter you can choose in main.py file, you also can change the parameters directly in main.py file.
for example:sudo python3 main.py --action="train" --choose_net="my_drsnet_A"

For test:
sudo python3 --action="test" --choose_net="NET"
for example:sudo python3 main.py --action="test" --choose_net="my_drsnet_A"

to change datasets:
first,download the add-rain datasets from the link below, and change the folder name of dataset in main.py

to get original UAS dataset:https://pan.baidu.com/s/1IWSVKYBrYwxaRThPfDsDGg
to get original BPS dataset:http://www.cbsr.ia.ac.cn/users/ynyu/dataset/

to get UAS-add-rain and BPS-add-rain datasets:
link:https://pan.baidu.com/s/1zRcBd2vTuKFWTIsQJrbQnA 
password: awfy

to get pretrained weights of different scales on UAS-ad-rain and BPS-ad-rain datasets:
link: https://pan.baidu.com/s/1Xv98jfHIJeyToV3PwfEJfQ  
password: 0tt9

the article related to this code is available here:https://link.springer.com/article/10.1007/s11554-020-01042-2?wt_mc=Internal.Event.1.SEM.ArticleAuthorOnlineFirst&utm_source=ArticleAuthorOnlineFirst&utm_medium=email&utm_content=AA_en_06082018&ArticleAuthorOnlineFirst_20201103

any problems, please do not hesitate to contact with me, my email is 2463908977@qq.com.
