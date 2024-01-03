import os
import glob
from osgeo import gdal, osr, ogr,gdalconst
import numpy as np
from pyproj import Proj
from scipy import ndimage

def S2tif(filename):
    # 打开栅格数据集
    print(filename)
    root_ds = gdal.Open(filename)
    print(type(root_ds))
    # 返回结果是一个list，list中的每个元素是一个tuple，每个tuple中包含了对数据集的路径，元数据等的描述信息
    # tuple中的第一个元素描述的是数据子集的全路径
    ds_list = root_ds.GetSubDatasets()  # 获取子数据集。该数据以数据集形式存储且以子数据集形式组织
    visual_ds = gdal.Open(ds_list[0][0])  # 打开第1个数据子集的路径。ds_list有4个子集，内部前段是路径，后段是数据信息
    visual_arr = visual_ds.ReadAsArray()  # 将数据集中的数据读取为ndarray

    # 创建.tif文件
    band_count = visual_ds.RasterCount  # 波段数
    xsize = visual_ds.RasterXSize
    ysize = visual_ds.RasterYSize
    out_tif_name = filename.split(".SAFE")[0] + ".tif"
    driver = gdal.GetDriverByName("GTiff")
    out_tif = driver.Create(out_tif_name, xsize, ysize, band_count, gdal.GDT_Float32)
    out_tif.SetProjection(visual_ds.GetProjection())  # 设置投影坐标
    out_tif.SetGeoTransform(visual_ds.GetGeoTransform())

    for index, band in enumerate(visual_arr):
        band = np.array([band])
        for i in range(len(band[:])):
            # 数据写出
            out_tif.GetRasterBand(index + 1).WriteArray(band[i])  # 将每个波段的数据写入内存，此时没有写入硬盘
    out_tif.FlushCache()  # 最终将数据写入硬盘
    out_tif = None  # 注意必须关闭tif文件
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

def srtm_select(x1,y1,x2,y2):
    srtm_list = []
    with open('../dem_SPARCS/lon_lat_srtm.txt', 'r+') as f:
        lon_lat_text = f.readlines()
    f.close()
    for a in range(len(lon_lat_text)):
        srtm_name, a1, b1, a2, b2 = lon_lat_text[a].split(' ')
        a1, b1, a2, b2 = np.float64(a1), np.float64(b1),np.float64(a2), np.float64(b2)
        #两点都包含
        if (x1 >= a1) and (y1 <= b1):
            if (x2 <= a2) and (y2 >= b2):
                srtm_list.append(srtm_name)
        #右下角在SRTM的右上方经度偏大，正常应该是左上方
        if (x1 >= a1) and (y1 <= b1):
            if(x2 >= a2) and (y2 >= b2):
                for b in range(len(lon_lat_text)):
                    srtm_name_right, a1_right, b1_right, a2_right, b2_right = lon_lat_text[a].split(' ')
                    a1_right, b1_right, a2_right, b2_right = np.float64(a1_right), np.float64(b1_right), \
                                                             np.float64(a2_right), np.float64(b2_right)
                    if (x2 <= a2_right) and (y2 >= b2_right):
                        srtm_list.append(srtm_name)
                        srtm_list.append(srtm_name_right)
        #右下角在SRTM的左下角，应找下方
        if (x1 >= a1) and (y1 <= b1):
            if (x2 <= a2) and (y2 <= b2):
                for b in range(len(lon_lat_text)):
                    srtm_name_b, a1_b, b1_b, a2_b, b2_b = lon_lat_text[a].split(' ')
                    a1_b, b1_b, a2_b, b2_b = np.float64(a1_b), np.float64(b1_b), \
                                            np.float64(a2_b), np.float64(b2_b)
                    if (x2 <= a2_b) and (y2 >= b2_b):
                        srtm_list.append(srtm_name)
                        srtm_list.append(srtm_name_right)
    return srtm_list


def geo2imagexy(dataset, x, y, x_geo, y_geo):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[90, 0], [0,-90]])
    b = np.array([x - x_geo, y - y_geo])
    return np.linalg.solve(a, b)

if __name__ == '__main__':
    #融合
    # root = '../data/WHUS2-CD/L1Cdata'
    # product_name_list = os.listdir(root)
    # data_list = []
    # for i in range(len(product_name_list)):
    #     SAFE_Path = (os.path.join(root,product_name_list[i]))
    #     tmp_list = glob.glob(SAFE_Path + "\\*.SAFE")
    #     data_list.append(tmp_list[0])
    #
    # for i in range(len(data_list)):
    #     data_path = data_list[i]
    #     filename = os.path.join(data_path,'MTD_MSIL1C.xml')
    #     S2tif(filename)
    #     print('done',i)

    #重采样
    # with open('WHU_SRTM_name_list.txt','r+') as f:
    #     tmp_list = f.readlines()
    # f.close()
    # whu_name_list = []
    # for i in range(len(tmp_list)):
    #     whu_name_list.append(tmp_list[i].split(' ')[0])
    # root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHU_30m\raw\mask'
    # data_name_list = whu_name_list
    # for i in range(len(data_name_list)):
    #     input_ds = gdal.Open(os.path.join(root,data_name_list[i]))
    #     inputProj = input_ds.GetProjection()
    #     name = data_name_list[i][:-4] +'_mask.tif'
    #     output_image_path = os.path.join(r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHU_90m\raw\mask',name)
    #     referencefileProj = input_ds.GetProjection()
    #     referencefileTrans = list(input_ds.GetGeoTransform())
    #     bandreferencefile = input_ds.GetRasterBand(1)
    #     x = 1220
    #     y = 1220
    #     nbands = 1
    #     referencefileTrans[1] *= 3
    #     referencefileTrans[5] *= 3
    #     driver = gdal.GetDriverByName('GTiff')
    #     output = driver.Create(output_image_path, x, y, nbands, bandreferencefile.DataType)
    #     output.SetGeoTransform(referencefileTrans)
    #     output.SetProjection(referencefileProj)
    #     # options = gdal.WarpOptions(srcSRS=inputProj, dstSRS=referencefileProj, resampleAlg=gdalconst.GRA_Bilinear)
    #     # gdal.Warp(output,input_ds,options=options)
    #     # print('donw,',i)
    #     data = input_ds.ReadRaster(
    #             buf_xsize=x, buf_ysize=y)
    #     output.WriteRaster(0, 0, x, y, data)
    #     output.FlushCache()
    #     for i in range(1):
    #         output.GetRasterBand(i + 1).ComputeStatistics(False)
    #
    #     output.BuildOverviews('average', [2, 4, 8, 16])

    # 加载投影信息
    # img_path = '../data/WHUS2-CD/WHU_10m'
    # mask_path = '../data/WHUS2-CD/ReferenceMask/ReferenceMask'
    # img_name_list = os.listdir(img_path)
    # mask_name_list = os.listdir(mask_path)
    # for i in range(len(img_name_list)):
    #     output_path = os.path.join('../data/WHUS2-CD/WHU_jiangcaiyang/mask_tmp',mask_name_list[i])
    #     img = gdal.Open(os.path.join(img_path,img_name_list[i]))
    #     label = gdal.Open(os.path.join(mask_path,mask_name_list[i]))
    #     geotrans = img.GetGeoTransform()
    #     projection = img.GetProjection()
    #     band1 = label.GetRasterBand(1)
    #     data_type = band1.DataType
    #     cols = img.RasterXSize
    #     rows = img.RasterYSize
    #     driver = gdal.GetDriverByName('GTiff')
    #     output = driver.Create(output_path,cols,rows,1,data_type)
    #     output.SetProjection(projection)
    #     output.SetGeoTransform(geotrans)
    #     output.GetRasterBand(1).WriteArray(label.ReadAsArray())
    #     del img,label,output
    #     print('done,',i)

    #查看whu数据集经纬度区间
    # root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHUS2-CD\WHU_jiangcaiyang\img'
    # img_name_list = os.listdir(root)
    # with open('lon_lat_whu.txt', 'a+')as f:
    #     for i in range(len(img_name_list)):
    #         img_name = img_name_list[i]
    #         img = gdal.Open(os.path.join(root, img_name_list[i]))
    #         top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(img)
    #         zone = img.GetProjection().split(' ')[5].split('"')[0][:-1]
    #         zone = int(zone)
    #         p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
    #         top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon, top_left_corner_lat, inverse=True)
    #         bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat,
    #                                                               inverse=True)
    #         f.writelines(img_name_list[i] + ' ' + str(top_left_corner_lon) + ' ' + str(top_left_corner_lat) + ' ' + str(
    #             bottom_right_corner_lon) + ' ' + str(bottom_right_corner_lat))
    #         f.write('\n')
    # f.close()

    # 对比经纬度获取对应srtm数据
    # root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHUS2-CD\WHU_jiangcaiyang\img'
    # img_name_list = os.listdir(root)
    # with open('../dem_SPARCS/lon_lat_srtm.txt', 'r+') as f:
    #     lon_lat_text = f.readlines()
    # f.close()
    # for i in range(len(img_name_list)):
    #     img = gdal.Open(os.path.join(root, img_name_list[i]))
    #     top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(img)
    #     zone = img.GetProjection().split(' ')[5].split('"')[0][:-1]
    #     zone = int(zone)
    #     p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
    #     top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon, top_left_corner_lat, inverse=True)
    #     bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat,
    #                                                           inverse=True)
    #     for j in range(len(lon_lat_text)):
    #         srtm_name, top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = \
    #         lon_lat_text[j].split(' ')
    #         top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = \
    #             np.float64(top_left_srtm_lon), np.float64(top_left_srtm_lat), np.float64(
    #                 bottom_right_srtm_lon), np.float64(bottom_right_srtm_lat)
    #         if (top_left_corner_lon >= top_left_srtm_lon) and (top_left_corner_lat <= top_left_srtm_lat):
    #             if (bottom_right_corner_lon <= bottom_right_srtm_lon) and (
    #                     bottom_right_corner_lat >= bottom_right_srtm_lat):
    #                 with open('WHU_SRTM_name_list.txt', 'a+') as f:
    #                     f.writelines(img_name_list[i] + ' ' + srtm_name)
    #                     f.write('\n')
    # f.close()

    #对比经纬度获取对应srtm数据，二次查询版
    # whu_root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHUS2-CD\WHU_jiangcaiyang\img'
    # SRTM_root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\SRTM_90米_DEM'
    # whu_name_list = os.listdir(whu_root)
    # SRTM_name_list = os.listdir(SRTM_root)
    # for i in range(len(whu_name_list)):
    #     whu_img = gdal.Open(os.path.join(whu_root, whu_name_list[i]))
    #     top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(whu_img)
    #     zone = whu_img.GetProjection().split(' ')[5].split('"')[0][:-1]
    #     zone = int(zone)
    #     p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
    #     top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon, top_left_corner_lat, inverse=True)
    #     bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat,inverse=True)
    #     srtm_list = srtm_select(top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat)
    #     with open('./WHU_SRTM_name_list.txt','a+')as f:
    #         f.write(whu_name_list[i] + ' ')
    #         for j in range(len(srtm_list)):
    #             f.write(srtm_list[j])
    #             f.write(' ')
    #         f.write('\n')
    # f.close()

    #切割dem
    dataset_path = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHU_30m\raw\img'
    srtm_path = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\SRTM_90米_DEM'
    with open('WHU_SRTM_name_list.txt', 'r+')as f:
        data2srtm_list = f.readlines()
    f.close()
    dataset_name_list = [x.split(' ')[0] for x in data2srtm_list]
    srtm_name_list = [x.split(' ')[1].split('\n')[0] for x in data2srtm_list]
    for i in range(len(dataset_name_list)):
        img = gdal.Open(os.path.join(dataset_path,dataset_name_list[i]))
        srtm = gdal.Open(os.path.join(srtm_path,srtm_name_list[i],(srtm_name_list[i] + '.tif')))
        top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(img)
        top_left_corner_lon_s, top_left_corner_lat_s, bottom_right_corner_lon_s, bottom_right_corner_lat_s = project_xy(srtm)
        zone = img.GetProjection().split(' ')[5].split('"')[0][:-1]
        zone = int(zone)
        p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
        x1_geo,y1_geo = p1(top_left_corner_lon_s, top_left_corner_lat_s)
        srtm_band = srtm.GetRasterBand(1).ReadAsArray()
        x1, y1 = geo2imagexy(srtm, top_left_corner_lon, top_left_corner_lat, x1_geo, y1_geo)
        x2, y2 = geo2imagexy(srtm, bottom_right_corner_lon, bottom_right_corner_lat, x1_geo, y1_geo)
        output = srtm_band[int(x1):int(x2), int(y1):int(y2)]
        img_name = dataset_name_list[i][:-4] + '_dem.tif'
        output_file = os.path.join(r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WHU_30m\raw\dem2',img_name)
        output = ndimage.zoom(output, (1220 / output.shape[0], 1220 / output.shape[1]), order=3, mode='nearest')
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_file, 1220, 1220, 1, gdal.GDT_UInt16)
        projection = img.GetProjection()
        dataset.SetProjection(projection)
        geotransform = list(img.GetGeoTransform())
        geotransform[1] *= 3
        geotransform[5] *= 3
        dataset.SetGeoTransform(geotransform)
        band = dataset.GetRasterBand(1)
        band.WriteArray(output)
        del dataset