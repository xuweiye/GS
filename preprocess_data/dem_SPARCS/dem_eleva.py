import os
from osgeo import gdal
from osgeo import gdal,osr
from pyproj import Proj
gdal.PushErrorHandler('CPLQuietErrorHandler')
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
def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[1], coords[0]

if __name__ == '__main__':
    # root = r'D:\software\baiduyundown\(A014)SRTM 90米 DEM Data'
    # srtm_name_list = os.listdir(root)
    # srtm_name_list = srtm_name_list[1:]
    # with open('lon_lat_srtm.txt','a+') as f:
    #     for i in range(len(srtm_name_list)):
    #         srtm_name = srtm_name_list[i] + '.tif'
    #         srtm_img = gdal.Open(os.path.join(root,srtm_name_list[i],srtm_name))
    #         top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(srtm_img)
    #         # print(top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat)
    #         f.writelines(srtm_name_list[i] + ' ' + str(top_left_corner_lon) + ' ' + str(top_left_corner_lat) + ' ' + str(bottom_right_corner_lon) + ' ' + str(bottom_right_corner_lat))
    #         f.write('\n')
    # f.close()

    root = '../data/SPARCS/'
    temp_list = os.listdir(root)
    img_name_list = [x for x in temp_list if x.endswith('data.tif')]
    with open('lon_lat_sparcs.txt', 'a+')as f:
        for i in range(len(img_name_list)):
            img_name = img_name_list[i] +'tif'
            img = gdal.Open(os.path.join(root,img_name_list[i]))
            top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(img)
            # top_left_corner_lon, top_left_corner_lat = geo2lonlat(img, top_left_corner_lon, top_left_corner_lat)
            # bottom_right_corner_lon, bottom_right_corner_lat = geo2lonlat(img, bottom_right_corner_lon,bottom_right_corner_lat)
            zone = img.GetProjection().split(' ')[5].split('"')[0][:-1]
            zone = int(zone)
            p1 = Proj(proj='utm',zone=zone,ellps='WGS84', preserve_units='m')
            top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon,top_left_corner_lat,inverse=True)
            bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat,inverse=True)
            f.writelines(img_name_list[i] + ' ' + str(top_left_corner_lon) + ' ' + str(top_left_corner_lat) + ' ' + str(bottom_right_corner_lon) + ' ' + str(bottom_right_corner_lat))
            f.write('\n')
    f.close()
