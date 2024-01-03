import os
from osgeo import gdal
from xml.dom import minidom
from pyproj import Proj
import numpy as np
from scipy import ndimage
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


def geo2imagexy(dataset, x, y, x_geo, y_geo):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[160, 0], [0,-160]])
    b = np.array([x - x_geo, y - y_geo])
    return np.linalg.solve(a, b)
# 查看whu数据集经纬度区间
# root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\mss'
# img_name_list = os.listdir(root)
# img_name_list = [x for x in img_name_list if x.endswith('.tiff')]
# xml_list = [x[:-5] + '.xml' for x in img_name_list]
# with open('lon_lat_wfv.txt', 'a+')as f:
#     for i in range(len(img_name_list)):
#         img = gdal.Open(os.path.join(root, img_name_list[i]))
#         xml_doc = minidom.parse(os.path.join(root, xml_list[i]))
#         top_left_corner_lon = xml_doc.getElementsByTagName('TopLeftLongitude')[0].firstChild.data
#         top_left_corner_lat = xml_doc.getElementsByTagName('TopLeftLatitude')[0].firstChild.data
#         bottom_right_corner_lon = xml_doc.getElementsByTagName('BottomRightLatitude')[0].firstChild.data
#         bottom_right_corner_lat = xml_doc.getElementsByTagName('BottomRightLongitude')[0].firstChild.data
#         print(top_left_corner_lat)
#         f.writelines(img_name_list[i] + ' ' + str(top_left_corner_lon) + ' ' + str(top_left_corner_lat) + ' ' + str(
#             bottom_right_corner_lon) + ' ' + str(bottom_right_corner_lat))
#         f.write('\n')
# f.close()

# 对比经纬度获取对应srtm数据
# root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\mss'
# img_name_list = os.listdir(root)
# img_name_list = [x for x in img_name_list if x.endswith('.tiff')]
# xml_list = [x[:-5] + '.xml' for x in img_name_list]
# with open('../dem_SPARCS/lon_lat_srtm.txt', 'r+',encoding='utf-8') as f:
#     lon_lat_text = f.readlines()
# f.close()
# for i in range(len(img_name_list)):
#     img = gdal.Open(os.path.join(root, img_name_list[i]))
#     xml_doc = minidom.parse(os.path.join(root, xml_list[i]))
#     top_left_corner_lon = np.float64(xml_doc.getElementsByTagName('TopLeftLongitude')[0].firstChild.data)
#     top_left_corner_lat = np.float64(xml_doc.getElementsByTagName('TopLeftLatitude')[0].firstChild.data)
#     bottom_right_corner_lon = np.float64(xml_doc.getElementsByTagName('BottomRightLatitude')[0].firstChild.data)
#     bottom_right_corner_lat = np.float64(xml_doc.getElementsByTagName('BottomRightLongitude')[0].firstChild.data)
#     for j in range(len(lon_lat_text)):
#         srtm_name, top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = \
#         lon_lat_text[j].split(' ')
#         top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = \
#             np.float64(top_left_srtm_lon), np.float64(top_left_srtm_lat), np.float64(
#                 bottom_right_srtm_lon), np.float64(bottom_right_srtm_lat)
#         if (top_left_corner_lon >= top_left_srtm_lon) and (top_left_corner_lat <= top_left_srtm_lat):
#             if (bottom_right_corner_lon <= bottom_right_srtm_lon) and (
#                     bottom_right_corner_lat >= bottom_right_srtm_lat):
#                 with open('WFV_SRTM_name_list.txt', 'a+') as f:
#                     f.writelines(img_name_list[i] + ' ' + srtm_name)
#                     f.write('\n')
# f.close()

# 切割dem
dataset_path = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\mss'
srtm_path = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\SRTM_90米_DEM'
with open('WFV_SRTM_name_list.txt', 'r+')as f:
    data2srtm_list = f.readlines()
f.close()
dataset_name_list = [x.split(' ')[0] for x in data2srtm_list]
srtm_name_list = [x.split(' ')[1].split('\n')[0] for x in data2srtm_list]
xml_list = [x[:-5] + '.xml' for x in dataset_name_list]
for i in range(len(dataset_name_list)):
    img = gdal.Open(os.path.join(dataset_path,dataset_name_list[i]))
    srtm = gdal.Open(os.path.join(srtm_path,srtm_name_list[i],(srtm_name_list[i] + '.tif')))
    xml_doc = minidom.parse(os.path.join(dataset_path, xml_list[i]))
    top_left_corner_lon = np.float64(xml_doc.getElementsByTagName('TopLeftLongitude')[0].firstChild.data)
    top_left_corner_lat = np.float64(xml_doc.getElementsByTagName('TopLeftLatitude')[0].firstChild.data)
    bottom_right_corner_lon = np.float64(xml_doc.getElementsByTagName('BottomRightLatitude')[0].firstChild.data)
    bottom_right_corner_lat = np.float64(xml_doc.getElementsByTagName('BottomRightLongitude')[0].firstChild.data)
    top_left_corner_lon_s, top_left_corner_lat_s, bottom_right_corner_lon_s, bottom_right_corner_lat_s = project_xy(srtm)
    zone = img.GetProjection().split(' ')[5].split('"')[0][:-1]
    zone = int(zone)
    p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
    x1_geo,y1_geo = p1(top_left_corner_lon_s, top_left_corner_lat_s)
    srtm_band = srtm.GetRasterBand(1).ReadAsArray()
    x1_geo = np.float64(xml_doc.getElementsByTagName('TopLeftMapX')[0].firstChild.data)
    y1_geo = np.float64(xml_doc.getElementsByTagName('TopLeftMapY')[0].firstChild.data)
    # x2 = np.float64(xml_doc.getElementsByTagName('BottomRightMapX')[0].firstChild.data)
    # y2 = np.float64(xml_doc.getElementsByTagName('BottomRightMapY')[0].firstChild.data)

    x1, y1 = geo2imagexy(srtm, top_left_corner_lon, top_left_corner_lat, x1_geo, y1_geo)
    x2, y2 = geo2imagexy(srtm, bottom_right_corner_lon, bottom_right_corner_lat, x1_geo, y1_geo)
    output = srtm_band[int(x1):int(x2), int(y1):int(y2)]
    img_name = dataset_name_list[i][:-5] + '_dem.tif'
    output_file = os.path.join(r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\dem',img_name)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_file, img.RasterXSize, img.RasterYSize, 1, gdal.GDT_UInt16)
    projection = img.GetProjection()
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(output)
    del dataset