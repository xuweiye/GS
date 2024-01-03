# data loader
from __future__ import print_function, division
import os
import torch
# from skimage import io, transform, color
import random
# from PIL import Image
import glob
import os
import cv2 as cv
import torch
import numpy as np
# from torchvision import models,transforms
from torch.utils.data import DataLoader
# import gdal
from osgeo import gdal
from pyproj import Proj
from tqdm import tqdm

gdal.PushErrorHandler('CPLQuietErrorHandler')


# ==========================dataset load==========================
class ToTensorNorm(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if np.max(image) != 0:
            image = image / np.max(image)
        else:
            image = image

        return image, label

def project_xy(dataset):
    geo_information = dataset.GetGeoTransform()
    col = dataset.RasterXSize  # 行数
    row = dataset.RasterYSize  # 列数
    # band = dataset.RasterCount  # 波段

    # 左上角经纬度方向投影坐标
    top_left_corner_lon = geo_information[0]
    top_left_corner_lat = geo_information[3]

    # 左下角经纬度方向投影坐标
    # bottom_left_corner_lon = geo_information[0] + row * geo_information[2]
    # bottom_left_corner_lat = geo_information[3] + row * geo_information[5]

    # 右上角经纬度方向投影坐标
    # top_right_corner_lon = geo_information[0] + col * geo_information[1]
    # top_right_corner_lat = geo_information[3] + col * geo_information[4]

    # 右下角经纬度方向投影坐标
    bottom_right_corner_lon = geo_information[0] + col * geo_information[1] + row * geo_information[2]
    bottom_right_corner_lat = geo_information[3] + col * geo_information[4] + row * geo_information[5]

    return top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat

# for input image
def toTensor(pic):
    if isinstance(pic, np.ndarray):
        # handle numpy array
        # print pic.shape
        pic = pic.transpose((2, 0, 1))
        # print pic.shape

        img = torch.from_numpy(pic.astype('float32'))
        # backward compatibility
        # for 8 bit images
        # return img.float().div(255.0)
        # for 16 bit images
        # return img.float().div(255.0)
        return img.float()


# def normalize(tensor,mean):
#     for t,m in zip(tensor,mean):
#         t.sub_(m)
#     return tensor

def maskToTensor(img):
    return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class SPARCS_MSS_CloudSnowDataset(object):
    # image: mss 16 bit
    # label: 0,1 8 bit

    def __init__(self,
                 rootPath='./data/LevirCS/crop//train',
                 joint_transform=None,model='train',shuffle=False):
        # bands
        """
        image 4 bands  -> 0-1023 ->  0-1
        dem 1 band   -> /10000.0 -> almost 0-1
        geo 2 bands  -> 1st long -180-180 -> 0-1 ; 2nd lat -90 - 90 -> 0-1
        time 1 band  -> 1.1-12.31  /365.0 or 366.0 ->  0-1
        """
        # self.type = type
        # self.demFlag = self.type/4
        # self.geoFlag = ((self.type%4)>1)*1
        # self.timeFlag = self.type%2
        mss_data_path = os.path.join(rootPath,'mss')
        dem_data_path = os.path.join(rootPath, 'dem')
        label_data_path = os.path.join(rootPath, 'mask')
        mss_name_list = sorted(os.listdir(mss_data_path))
        dem_name_list = sorted(os.listdir(dem_data_path))
        label_name_list = sorted(os.listdir(label_data_path))
        # txtFile = open(trainTxtPath, 'r')
        self.imgPath = []
        self.demPath = []
        self.seglabelPath = []
        if shuffle:
            combined_list = list(zip(mss_name_list, dem_name_list, label_name_list))
            random.shuffle(combined_list)
            total_samples = len(combined_list)
            train_samples = int(total_samples * 0.8)
        if model == 'train':
            train_list = combined_list[:train_samples]
            mss_name_list, dem_name_list, label_name_list = zip(*train_list)
        if model == 'val':
            val_list = combined_list[train_samples:]
            mss_name_list, dem_name_list, label_name_list = zip(*val_list)

        for i in range(len(dem_name_list)):
            self.imgPath.append(os.path.join(mss_data_path,mss_name_list[i]))
            self.demPath.append(os.path.join(dem_data_path,dem_name_list[i]))
            self.seglabelPath.append(os.path.join(label_data_path,label_name_list[i]))

        # print len(self.img)
        # print len(self.seglabel)
        # print len(self.scenelabel)

        self.joint_transform = joint_transform


    def __getitem__(self, index):

        # img = self.imgs[index]
        imgPath_s = self.imgPath[index]
        demPath_s = self.demPath[index]
        seglabelPath_s = self.seglabelPath[index]
        # #读取时间
        # with open('../data/SPARCS/SPARCS_Time.txt','r') as file:
        #     # 打开txt文件并读取其内容
        #     content = file.read()
        #     # 使用splitlines()方法将每一行分割成单独的元素
        # lines = content.splitlines()
        # # 将每个元素（即每行文本）添加到一个列表中
        # your_list = lines
        # for i in range(len(your_list)):
        #     if imgPath_s.split('\\').split('_')[0] == your_list[i].split(' ')[0]:
        #         time = your_list[i].split(' ')[1]
        # # 打印列表以检查内容
        # print(your_list)
        imgDataset = gdal.Open(imgPath_s)
        imgGeotransform = list(imgDataset.GetGeoTransform())
        tmp = imgDataset.GetRasterBand(1).ReadAsArray()
        top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(imgDataset)
        zone = imgDataset.GetProjection().split(' ')[5].split('"')[0][:-1]
        zone = int(zone)
        p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
        top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon, top_left_corner_lat, inverse=True)
        bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat, inverse=True)
        imgGeotransform[0],imgGeotransform[3] = top_left_corner_lon, top_left_corner_lat
        imgGeotransform[1] = (bottom_right_corner_lon - top_left_corner_lon)/224
        imgGeotransform[5] = (top_left_corner_lat - bottom_right_corner_lat)/224


        datacube = np.zeros((tmp.shape[0], tmp.shape[1], 7)).astype('float32')

        for i in range(0, 4):
            datacube[:, :, i] = imgDataset.GetRasterBand(i + 2).ReadAsArray()
            # 大于1023设1023为最大值，小于1023保持原值，归一化
            # datacube[:, :, i] = (datacube[:, :, i] > 1023) * 1023 + (datacube[:, :, i] <= 1023) * datacube[:, :, i]
            datacube[:, :, i] = datacube[:, :, i] / np.max(datacube[:,:,i])
        # imgDataset=None
        if np.max(datacube) != 0:
            datacube = datacube/np.max(datacube)
        del imgDataset

        demDataset = gdal.Open(demPath_s)
        dem_data = demDataset.GetRasterBand(1).ReadAsArray()
        if np.max(dem_data) != 0:
            datacube[:, :, 4] = ((demDataset.GetRasterBand(1).ReadAsArray()).astype('float32')) / np.max(dem_data)
        # demDataset = None
        del demDataset,dem_data


        datacube[:, :, 5] = (imgGeotransform[0] + imgGeotransform[1] * np.tile(np.arange(tmp.shape[1]),(tmp.shape[0], 1)) +
                             imgGeotransform[2] * ((np.ones((tmp.shape[1], tmp.shape[0])) * np.arange(tmp.shape[0])).transpose()) + 180.0) / 360.0
        datacube[:, :, 6] = (imgGeotransform[3] + imgGeotransform[4] * np.tile(np.arange(tmp.shape[1]),(tmp.shape[0], 1)) +
                             imgGeotransform[5] * ((np.ones((tmp.shape[1], tmp.shape[0])) * np.arange(tmp.shape[0])).transpose()) + 90.0) / 180.0

        # seglabel = cv.imread(seglabelPath_s, 0)
        seglabel = gdal.Open(seglabelPath_s).ReadAsArray()
        seglabel[seglabel != 5] = 0
        seglabel[seglabel==5] = 1
        datacube[datacube<0] = 0

        datacube = datacube[:,:,[0,1,2,3,5]]
        datacube = toTensor(datacube)

        seglabel = maskToTensor(seglabel)
        seglabel = torch.unsqueeze(seglabel, 0)

        if self.joint_transform is not None:
            datacube, seglabel = self.joint_transform(datacube, seglabel)

        return datacube,seglabel
        # return datacube, seglabel

    def __len__(self):
        return len(self.seglabelPath)

# if __name__ == '__main__':
    # 查询有无脏数据
    # dataset = SPARCS_MSS_CloudSnowDataset('../data/SPARCS/SPARCS_Crop/train',None,None)
    # c = 0
    # for i in tqdm(range(dataset.__len__())):
    #     k = c
    #     mss , label = dataset.__getitem__(i)
    #     sum0 = torch.sum(label == 0)
    #     sum1 = torch.sum(label == 1)
    #     print(sum0)
    #     print(sum1)
    #     if sum0 + sum1 != 224 * 224:
    #         c = c+1
    #         print(sum1+sum0)
    #     if k !=c:
    #         print(c)
    # print(c)





    # root = '../data/SPARCS/SPARCS_Crop/train/'
    # mss_root = os.path.join(root,'mss')
    # mask_root = os.path.join(root, 'mask')
    # dem_root = os.path.join(root, 'dem')
    # mss_list = os.listdir(mss_root)
    # mask_list = os.listdir(mask_root)
    # dem_list = os.listdir(dem_root)
    # for i in range(len(mask_list)):
    #     mss = gdal.Open(os.path.join(mss_root, mss_list[i]))
    #     for i in range(0, 4):
    #         img = mss.GetRasterBand(i + 1).ReadAsArray()
    #         if np.max(img)>65535.0:
    #             print(np.max(img))
            # img = img / 65535.0

    # for i in range(len(mask_list)):
    #     mss = gdal.Open(os.path.join(mss_root,mss_list[i]))
    #     imgGeotransform = list(mss.GetGeoTransform())
    #     img = mss.ReadAsArray()
    #     top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(mss)
    #     zone = mss.GetProjection().split(' ')[5].split('"')[0][:-1]
    #     zone = int(zone)
    #     p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
    #     top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon, top_left_corner_lat, inverse=True)
    #     bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat,
    #                                                           inverse=True)
    #     imgGeotransform[0], imgGeotransform[3] = top_left_corner_lon, top_left_corner_lat
    #     imgGeotransform[1] = (bottom_right_corner_lon - top_left_corner_lon) / 224
    #     imgGeotransform[5] = (top_left_corner_lat - bottom_right_corner_lat) / 224
    #     a,b = [],[]
    #     a = (imgGeotransform[0] + imgGeotransform[1] * np.tile(np.arange(img.shape[1]),(img.shape[0], 1)) +
    #                          imgGeotransform[2] * ((np.ones((img.shape[1], img.shape[0])) * np.arange(img.shape[0])).transpose()) + 180.0) / 360.0
    #     b = (imgGeotransform[3] + imgGeotransform[4] * np.tile(np.arange(img.shape[1]),(img.shape[0], 1)) +
    #                          imgGeotransform[5] * ((np.ones((img.shape[1], img.shape[0])) * np.arange(img.shape[0])).transpose()) + 90.0) / 180.0
    #     if np.max(a)>1 or np.max(b)>1:
    #         print(np.max(a))
    #         print(np.max(b))

    # root = '../data/SPARCS/SPARCS_Crop/test/'
    # dem_root = os.path.join(root, 'dem')
    # dem_list = os.listdir(dem_root)
    # max_data = 0
    # for i in range(len(dem_list)):
    #     dem = gdal.Open(os.path.join(dem_root,dem_list[i]))
    #     data = dem.ReadAsArray()
    #     if np.max(data) > max_data:
    #         max_data = np.max(data)
    #     print(max_data)
#11196

# DataLoader中collate_fn使用

def dataset_collate(batch):
    images = []
    pngs = []

    for img, png in batch:
        images.append(img)
        pngs.append(png)
        # seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    # seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs

