import os
import random
import torch
import numpy as np
from osgeo import gdal
from tqdm import tqdm

def extract_collapsed_cls(mask):
    mask[mask != 255] = 0#NoneCloud = 0 注意顺序！！！！
    mask[mask == 255] = 1# cloud.
    return mask

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


class WFV_MSS_CloudSnowDataset(object):
    # image: mss 16 bit
    # label: 0,1 8 bit

    def __init__(self,
                 rootPath='./data/WFV/crop/train',
                 joint_transform=None,model='train',shuffle=True):
        # bands
        """
        image 4 bands  -> 0-1023 ->  0-1
        dem 1 band   -> /10000.0 -> almost 0-1
        geo 2 bands  -> 1st long -180-180 -> 0-1 ; 2nd lat -90 - 90 -> 0-1
        time 1 band  -> 1.1-12.31  /365.0 or 366.0 ->  0-1
        """

        # only for train
        """
        mean_value: 257 #514
        mean_value: 268 #536
        mean_value: 276 #552
        mean_value: 238 #475
        """
        # self.type = type
        # self.demFlag = self.type/4
        # self.geoFlag = ((self.type%4)>1)*1
        # self.timeFlag = self.type%2
        mss_data_path = os.path.join(rootPath,'mss')
        # dem_data_path = os.path.join(rootPath, 'dem')
        label_data_path = os.path.join(rootPath, 'mask')
        mss_name_list = sorted(os.listdir(mss_data_path))
        # dem_name_list = sorted(os.listdir(dem_data_path))
        label_name_list = sorted(os.listdir(label_data_path))
        # txtFile = open(trainTxtPath, 'r')
        self.imgPath = []
        self.demPath = []
        self.seglabelPath = []
        if shuffle:
            # combined_list = list(zip(mss_name_list, dem_name_list, label_name_list))
            combined_list = list(zip(mss_name_list, label_name_list))
            random.shuffle(combined_list)
            total_samples = len(combined_list)
            train_samples = int(total_samples * 0.8)
        if model == 'train':
            train_list = combined_list[:train_samples]
            mss_name_list, label_name_list = zip(*train_list)
        if model == 'val':
            val_list = combined_list[train_samples:]
            mss_name_list,label_name_list = zip(*val_list)

        for i in range(len(mss_name_list)):
            self.imgPath.append(os.path.join(mss_data_path,mss_name_list[i]))
            # self.demPath.append(os.path.join(dem_data_path,dem_name_list[i]))
            self.seglabelPath.append(os.path.join(label_data_path,label_name_list[i]))

        # print len(self.img)
        # print len(self.seglabel)
        # print len(self.scenelabel)

        self.joint_transform = joint_transform

    def __getitem__(self, index):

        # img = self.imgs[index]
        imgPath_s = self.imgPath[index]
        # demPath_s = self.demPath[index]
        seglabelPath_s = self.seglabelPath[index]
        img_time = int(imgPath_s.split('\\')[-1].split('_')[-4])
        imgDataset = gdal.Open(imgPath_s)
        imgGeotransform = imgDataset.GetGeoTransform()
        # tmp = imgDataset.GetRasterBand(1).ReadAsArray()
        datacube = np.zeros((384, 384, 4)).astype('float32')
        # print(datacube.shape)
        # print(imgDataset.GetRasterBand(1).ReadAsArray().shape)


        for i in range(0, 4):
            tmp_data = imgDataset.GetRasterBand(i + 1).ReadAsArray()
            datacube[:, :, i] = tmp_data
        if np.max(datacube) != 0:
            datacube = datacube/np.max(datacube)
        # imgDataset=None
        datacube[datacube<0] = 0
        del imgDataset

        # demDataset = gdal.Open(demPath_s)
        # tmp_data = demDataset.ReadAsArray()
        # datacube[:, :, 4] = ((demDataset.GetRasterBand(1).ReadAsArray()).astype('float32'))
        # if np.max(datacube[:, :, 4]) != 0:
        #     datacube[:, :, 4] = datacube[:, :, 4] / np.max(datacube[:, :, 4])


        # demDataset = None
        # del demDataset

        # datacube[:, :, 5] = (imgGeotransform[0] + imgGeotransform[1] * np.tile(np.arange(tmp.shape[1]),
        #                      (tmp.shape[0], 1)) + imgGeotransform[2] * ((np.ones((tmp.shape[1], tmp.shape[0])) * np.arange(tmp.shape[0])).transpose()) + 180.0) / 360.0
        # datacube[:, :, 6] = (imgGeotransform[3] + imgGeotransform[4] * np.tile(np.arange(tmp.shape[1]),
        #                      (tmp.shape[0], 1)) + imgGeotransform[5] * ((np.ones((tmp.shape[1], tmp.shape[0])) * np.arange(tmp.shape[0])).transpose()) + 90.0) / 180.0
        # datacube[datacube<0] = 0
        # seglabel = cv.imread(seglabelPath_s, 0)
        seglabel = gdal.Open(seglabelPath_s).ReadAsArray()
        seglabel = extract_collapsed_cls(seglabel)

        datacube = toTensor(datacube)
        seglabel = maskToTensor(seglabel)
        seglabel = torch.unsqueeze(seglabel, 0)
        datacube = datacube[0:4,:,:]

        if self.joint_transform is not None:
            datacube, seglabel = self.joint_transform(datacube, seglabel)

        # then transform the input image to tensor
        return datacube, seglabel

    def __len__(self):
        return len(self.seglabelPath)


def custom_collate_fn(self, batch):
    # 过滤掉为None的图像
    batch = [item for item in batch if item is not None]
    return torch.stack(batch)

# if __name__ == '__main__':
#     dataset = WFV_MSS_CloudSnowDataset(rootPath=r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw', joint_transform=None,model='train')
#     for i in tqdm(range(dataset.__len__())):
#         data, label = dataset.__getitem__(i)
#     my_loader = CustomDataLoader(mydata,batch_size=2,num_workers=0,drop_last=True,shuffle=True,pin_memory=True)
#     print(my_loader)