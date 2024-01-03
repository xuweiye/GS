from osgeo import gdal
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split



def random_xy(x_size,y_size,split_window):
    np.random.seed(100)
    array_size = (10,1)
    x_size = x_size - split_window
    y_size = y_size - split_window
    x_array = np.random.randint(0,x_size, size=array_size)
    y_array = np.random.randint(0, y_size, size=array_size)
    return np.concatenate((x_array,y_array),axis=1)
def reguler_xy(x_size,y_size,split_window):
    x_size = x_size - split_window
    y_size = y_size - split_window
    x_array = np.arange(0, x_size + 1, split_window - split_window / 4).reshape(-1, 1)
    y_array = np.arange(0, y_size + 1, split_window - split_window / 4).reshape(-1, 1)
    cor = np.ones((len(x_array)*len(y_array),2))
    for i in range(len(y_array)):
        for j in range(len(x_array)):
            cor[i*len(x_array)+j,0] = x_array[j]
            cor[i*len(y_array)+j,1] = y_array[i]
    return cor

def patching_Geodata(img,dem,label,split_window,overlap_size,img_path,dem_path,label_path,coordinate_array,x_padding_size,y_padding_size):
    Geodata = img
    img = Geodata.ReadAsArray()
    if len(img.shape) == 3:
        img_geotrans = Geodata.GetGeoTransform()  # 获取仿射变换
        img_proj = Geodata.GetProjection()#获取地图投影信息
        img_datatype = Geodata.GetRasterBand(1).DataType
        im_bands, im_height, im_width = img.shape
    label_data = label.ReadAsArray()
    # dem_data = dem.ReadAsArray()
    # dem_datatype = dem.GetRasterBand(1).DataType
    npad = ((0, 0), (x_padding_size//2, x_padding_size//2), (y_padding_size//2, y_padding_size//2))
    npad_1 = ((x_padding_size//2, x_padding_size//2), (y_padding_size//2, y_padding_size//2))
    img = np.pad(img, pad_width=npad, mode='symmetric')
    label_data = np.pad(label_data, pad_width=npad_1, mode='symmetric')
    # dem_data = np.pad(dem_data, pad_width=npad_1, mode='symmetric')
    label_datatype = label.GetRasterBand(1).DataType

    for i in range(len(coordinate_array)):
        xmin = coordinate_array[i,0]
        ymin = coordinate_array[i,1]
        xmax = xmin + split_window
        ymax = ymin + split_window
        new_x_geo = img_geotrans[0] + xmin * img_geotrans[1] + ymin * img_geotrans[2]  # 新横坐标起始量
        new_y_geo = img_geotrans[3] + ymin * img_geotrans[5] + xmin * img_geotrans[4]  # 新纵坐标起始量
        new_geotransform = (new_x_geo, img_geotrans[1], img_geotrans[2], new_y_geo, img_geotrans[4], img_geotrans[5])
        xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
        img_patched = img[:,xmin:xmax, ymin:ymax]
        # dem_patched = dem_data[xmin:xmax, ymin:ymax]
        label_patched = label_data[xmin:xmax, ymin:ymax]
        #mss
        driver = gdal.GetDriverByName("GTiff")
        filename = img_path[:-5] + '_mss_patch-%d'% i +'.tif'
        dataset = driver.Create(filename, split_window, split_window, im_bands, img_datatype)
        dataset.SetGeoTransform(new_geotransform)  # 写入仿射变换参数
        dataset.SetProjection(img_proj)
        for k in range(im_bands):
            dataset.GetRasterBand(k + 1).WriteArray(img_patched[k])
        del dataset,driver
        # #dem
        # driver = gdal.GetDriverByName("GTiff")
        # filename = dem_path[:-8] + '_dem_patch-%d'% i +'.tif'
        # dataset = driver.Create(filename, split_window, split_window, 1, dem_datatype)
        # dataset.SetGeoTransform(new_geotransform)  # 写入仿射变换参数
        # dataset.SetProjection(img_proj)
        # dataset.GetRasterBand(1).WriteArray(dem_patched)
        # del dataset,driver
        #label
        driver = gdal.GetDriverByName("GTiff")
        filename = label_path[:-5] + '_label_patch-%d' % i + '.tif'
        dataset = driver.Create(filename, split_window, split_window, 1, label_datatype)
        # dataset.SetGeoTransform(new_geotransform)  # 写入仿射变换参数
        # dataset.SetProjection(img_proj)
        dataset.GetRasterBand(1).WriteArray(label_patched)
        del dataset,driver


def WFV_patch(opt):
    random.seed(101)
    raw_data_path = opt.data_dir_WFV
    crop_data_path = opt.data_crop_dir_WFV
    img_list = os.listdir('/data/wyxu/data/WFV/raw/mss')
    img_list = [x for x in img_list if x.endswith('.tiff')]
    print(img_list)
    if opt.train == 'train':
        crop_label_path = os.path.join(crop_data_path, 'train/mask/')
        crop_mss_path = os.path.join(crop_data_path, 'train/mss/')
        crop_dem_path = os.path.join(crop_data_path,'train/dem/')
        mss_name_list, _ = train_test_split(img_list, test_size=0.2)
    elif opt.train == 'test':
        crop_label_path = os.path.join(crop_data_path, 'test/mask/')
        crop_mss_path = os.path.join(crop_data_path, 'test/mss/')
        crop_dem_path = os.path.join(crop_data_path, 'test/dem/')
        _, mss_name_list = train_test_split(img_list, test_size=0.2)


    dem_name_list = [x[:-5]+'_dem.tif' for x in mss_name_list]
    label_name_list = mss_name_list
    spilt_window = opt.split_window
    patch_size = opt.patch_size
    overlap_size = opt.overlapping_size
    ##384 n - (n-1) 96 = 1500+pad,n=5 pad = 36
    for i in tqdm(range(len(dem_name_list))):
        mss_data = gdal.Open(os.path.join(raw_data_path,'mss',mss_name_list[i]))
        label_data = gdal.Open(os.path.join(raw_data_path,'mask', label_name_list[i]))
        dem_data = gdal.Open(os.path.join(raw_data_path,'dem', dem_name_list[i]))
        x_size = mss_data.RasterXSize
        y_size = mss_data.RasterYSize
        x_padding_size = 36
        y_padding_size = 36
        # datacube, seglabel = GeoEnigen(mss_data,label_data,dem_data)
        coordinate_array = reguler_xy(x_size+x_padding_size,y_size+y_padding_size,opt.split_window)

        img_path = os.path.join(crop_mss_path,mss_name_list[i])
        dem_path = os.path.join(crop_dem_path,dem_name_list[i])
        label_path = os.path.join(crop_label_path,label_name_list[i])

        patching_Geodata(mss_data,dem_data,label_data,spilt_window, overlap_size,img_path,dem_path,label_path,coordinate_array,x_padding_size,y_padding_size)

