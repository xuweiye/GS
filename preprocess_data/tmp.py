import os
from PIL import Image
import numpy as np

root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\result_img\LevirCS\crop\mask'
img_list = os.listdir(root)
save_root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\result_img\LevirCS\crop\mask_255'

for i in range(len(img_list)):
    img = Image.open(os.path.join(root,img_list[i]))
    img = np.array(img)
    img[img==128] = 255
    img = Image.fromarray(img).convert('L')
    img.save(os.path.join(save_root,img_list[i]))
