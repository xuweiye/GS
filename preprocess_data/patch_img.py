from osgeo import gdal
import os
import numpy as np
from tqdm import tqdm

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
    x_array = np.arange(0, x_size+1, split_window-split_window/4).reshape(-1,1)
    y_array = np.arange(0, y_size+1, split_window-split_window/4).reshape(-1,1)
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
    dem_data = dem.ReadAsArray()
    dem_datatype = dem.GetRasterBand(1).DataType
    npad = ((0, 0), (x_padding_size//2, x_padding_size//2), (y_padding_size//2, y_padding_size//2))
    npad_1 = ((x_padding_size//2, x_padding_size//2), (y_padding_size//2, y_padding_size//2))
    img = np.pad(img, pad_width=npad, mode='symmetric')
    label_data = np.pad(label_data, pad_width=npad_1, mode='symmetric')
    dem_data = np.pad(dem_data, pad_width=npad_1, mode='symmetric')
    label_datatype = label.GetRasterBand(1).DataType
    # img_shape = np.shape(img)  # 1340x1200,channels
    # stride = split_window

    # padding_size_x = split_window - (img_shape[1] % split_window)
    # padding_size_y = split_window - (img_shape[2] % split_window)
    # npad = ((0,0),(padding_size_x//2-1, padding_size_x//2-1), (padding_size_y//2-1, padding_size_y//2-1))
    # img = np.pad(img, pad_width=npad, mode='symmetric')

    # img_shape = np.shape(img)
    # img_padded = np.zeros((img_shape[0],img_shape[1] + 1, img_shape[2] + 1))
    # img_padded[:,1:1 + img_shape[1], 1:1 + img_shape[2]] = img

    # padding_size = int(split_window / 4)
    # img_padded = np.zeros((img_shape[0],img_shape[1] + 2 * padding_size, img_shape[2] + 2 * padding_size))
    # img_padded[:,96:96 + img_shape[1], 96:96 + img_shape[2]] = img

    # Find number of patches
    # n_width = int((np.size(img_padded, axis=0) - patch_size) / (patch_size - overlap))
    # n_height = int((np.size(img_padded, axis=1) - patch_size) / (patch_size - overlap))
    # n_width = int(np.size(img_padded, axis=1) / (split_window)) - 1  # 6
    # n_height = int(np.size(img_padded, axis=2) / (split_window)) - 1

    # Now cut into patches
    # n_bands = np.size(img_padded, axis=0)
    # img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=img.dtype)
    # img_patched = np.zeros((n_bands, split_window, split_window), dtype=img.dtype)

    # for i in range(0, n_width):
    #     for j in range(0, n_height):
    #         id = n_height * i + j

            # 重叠切割
            # xmin = split_window * i - i * stride
            # xmax = split_window * i + split_window - i * stride
            # ymin = split_window * j - j * stride
            # ymax = split_window * j + split_window - j * stride
            #非重叠切割
            # xmin = split_window * i
            # xmax = split_window * i + split_window
            # ymin = split_window * j
            # ymax = split_window * j + split_window

            # Cut out the patches.
            # img_patched[id, width , height, depth]
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
        dem_patched = dem_data[xmin:xmax, ymin:ymax]
        label_patched = label_data[xmin:xmax, ymin:ymax]
        #mss
        driver = gdal.GetDriverByName("GTiff")
        filename = img_path[:-9] + '_mss_patch-%d'% i +'.tif'
        dataset = driver.Create(filename, split_window, split_window, im_bands, img_datatype)
        dataset.SetGeoTransform(new_geotransform)  # 写入仿射变换参数
        dataset.SetProjection(img_proj)
        for k in range(im_bands):
            dataset.GetRasterBand(k + 1).WriteArray(img_patched[k])
        del dataset,driver
        #dem
        driver = gdal.GetDriverByName("GTiff")
        filename = dem_path[:-9] + '_dem_patch-%d'% i +'.tif'
        dataset = driver.Create(filename, split_window, split_window, 1, dem_datatype)
        dataset.SetGeoTransform(new_geotransform)  # 写入仿射变换参数
        dataset.SetProjection(img_proj)
        dataset.GetRasterBand(1).WriteArray(dem_patched)
        del dataset,driver
        #label
        driver = gdal.GetDriverByName("GTiff")
        # if LevirCS
        filename = label_path[:-11] + '_label_patch-%d' % i + '.tif'
        # if SPARCS

        dataset = driver.Create(filename, split_window, split_window, 1, label_datatype)
        # dataset.SetGeoTransform(new_geotransform)  # 写入仿射变换参数
        # dataset.SetProjection(img_proj)
        dataset.GetRasterBand(1).WriteArray(label_patched)
        del dataset,driver

            # img_patched[id,:, :, :] = img_padded[:,xmin:xmax, ymin:ymax]

    # return img_patched  # n_height and n_width are necessary for stitching image back together



# def GeoEnigen(mss_data,label_data,dem_data):
#     imgGeotransform = mss_data.GetGeoTransform()
#     tmp = mss_data.GetRasterBand(1).ReadAsArray()
#     float32 这些格式是文件最终大小的关键
    # datacube = np.zeros((tmp.shape[0], tmp.shape[1], 7)).astype('float32')

    # for i in range(0, 4):
    #     datacube[:, :, i] = mss_data.GetRasterBand(i + 1).ReadAsArray()
    #     大于1023设1023为最大值，小于1023保持原值，归一化
    #     datacube[:, :, i] = (datacube[:, :, i] > 1023) * 1023 + (datacube[:, :, i] <= 1023) * datacube[:, :, i]
    #     datacube[:, :, i] = datacube[:, :, i] / 1023.0
    # imgDataset=None
    # del mss_data
    #
    # datacube[:, :, 4] = ((dem_data.GetRasterBand(1).ReadAsArray()).astype('float32')) / 10000.0
    # demDataset = None
    # del dem_data
    #
    # datacube[:, :, 5] = (imgGeotransform[0] + imgGeotransform[1] * np.tile(np.arange(tmp.shape[1]),
    #                     (tmp.shape[0], 1)) + imgGeotransform[2] * ((np.ones((tmp.shape[1], tmp.shape[0])) * np.arange(tmp.shape[0])).transpose()) + 180.0) / 360.0
    # datacube[:, :, 6] = (imgGeotransform[3] + imgGeotransform[4] * np.tile(np.arange(tmp.shape[1]),
    #                     (tmp.shape[0], 1)) + imgGeotransform[5] * ((np.ones((tmp.shape[1], tmp.shape[0])) * np.arange(tmp.shape[0])).transpose()) + 90.0) / 180.0
    #
    # seglabel = label_data.ReadAsArray()
    # seglabel = seglabel[:,:,np.newaxis]
    # return datacube,seglabel

def LevirCS_patch(opt):
    raw_data_path = opt.data_dir_L
    crop_data_path = opt.data_crop_dir_L
    if opt.train == 'train':
        crop_label_path = os.path.join(crop_data_path, 'train/mask/')
        crop_mss_path = os.path.join(crop_data_path, 'train/mss/')
        crop_dem_path = os.path.join(crop_data_path,'train/dem/')
        raw_label_path = os.path.join(raw_data_path, 'train/mask/')
        raw_mss_path = os.path.join(raw_data_path, 'train/mss/')
        raw_dem_path = os.path.join(raw_data_path, 'train/dem/')
    elif opt.train == 'val':
        crop_label_path = os.path.join(crop_data_path, 'val/mask/')
        crop_mss_path = os.path.join(crop_data_path, 'val/mss/')
        crop_dem_path = os.path.join(crop_data_path, 'val/dem/')
        raw_label_path = os.path.join(raw_data_path, 'val/mask/')
        raw_mss_path = os.path.join(raw_data_path, 'val/mss/')
        raw_dem_path = os.path.join(raw_data_path, 'val/dem/')
    elif opt.train == 'test':
        crop_label_path = os.path.join(crop_data_path, 'test/mask/')
        crop_mss_path = os.path.join(crop_data_path, 'test/mss/')
        crop_dem_path = os.path.join(crop_data_path, 'test/dem/')
        raw_label_path = os.path.join(raw_data_path, 'test/mask/')
        raw_mss_path = os.path.join(raw_data_path, 'test/mss/')
        raw_dem_path = os.path.join(raw_data_path, 'test/dem/')

    mss_name_list = os.listdir(raw_mss_path)
    dem_name_list = os.listdir(raw_dem_path)
    label_name_list = os.listdir(raw_label_path)
    spilt_window = opt.split_window
    patch_size = opt.patch_size
    overlap_size = opt.overlapping_size
    for i in tqdm(range(len(dem_name_list))):
        mss_data = gdal.Open(os.path.join(raw_mss_path,mss_name_list[i]))
        label_data = gdal.Open(os.path.join(raw_label_path, label_name_list[i]))
        dem_data = gdal.Open(os.path.join(raw_dem_path, dem_name_list[i]))
        a = mss_data.RasterXSize
        b = mss_data.RasterYSize
        x_size = b
        y_size = a
        x_padding_size = opt.split_window - x_size%opt.split_window
        y_padding_size = opt.split_window - y_size % opt.split_window
        # datacube, seglabel = GeoEnigen(mss_data,label_data,dem_data)
        coordinate_array = reguler_xy(x_size+x_padding_size,y_size+y_padding_size,opt.split_window)

        img_path = os.path.join(crop_mss_path,mss_name_list[i])
        dem_path = os.path.join(crop_dem_path,dem_name_list[i])
        label_path = os.path.join(crop_label_path,label_name_list[i])

        patching_Geodata(mss_data,dem_data,label_data,spilt_window, overlap_size,img_path,dem_path,label_path,coordinate_array,x_padding_size,y_padding_size)
        # label_patched = patching_data(label_data, spilt_window, overlap_size,coordinate_array)


        # for patch in range(np.size(dem_patched, axis=0)):
            # if np.all(x_patched[patch, :, :, :]) != 0:  # 先不识别云影
            # img = np.zeros((384,384,10))
            # img = x_patched[patch, :, :, :]
            # img = PIL.Image.fromarray(img)
            # img.save(processed_data_path + 'train/img/' + product[:-9] + '_x_patch-%d'% patch)
            # np.save(crop_mss_path  + dem_name_list[i][:-9] + '_mss_patch-%d'% patch, mss_patched[patch, :, :, :])
            # label = np.zeros((384,384,1))
            # label = y_patched[patch, :, :, :]
            # label = PIL.Image.fromarray(label).convert('L')
            # label.save(processed_data_path + 'train/label/' + product[:-9] + '_y_patch-%d'% patch)
            # np.save(crop_label_path  + dem_name_list[i][:-11] + '_label_patch-%d'% patch, label_patched[patch, :, :])
            # np.save(crop_dem_path  + dem_name_list[i][:-11] + '_label_patch-%d'% patch, label_patched[patch, :, :])
        # print('NO.%d image is done '% i)

