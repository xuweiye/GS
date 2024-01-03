import os
from osgeo import gdal,osr
import numpy as np
import math
from pyproj import Proj,Transformer
from dem_eleva import project_xy
gdal.PushErrorHandler('CPLQuietErrorHandler')
#UTM - WGS
def transform_utm_into_lat_lon(x, y, zone=19):
    # verify the hemisphere
    # h_north = False
    # h_south = False
    # if (hemisphere == 'N'):
    #     h_north = True
    # elif (hemisphere == 'S'):
    #     h_south = True
    # else:
    #     print("Unknown hemisphere: " + hemisphere)

    proj_in = Proj(proj='utm', zone=zone, ellps='WGS84',preserve_units='m')

    lon, lat = proj_in(x, y,inverse=True)

    # just printing the floating point number with 6 decimal points will round it
    lon = math.floor(lon * 1000000) / 1000000
    lat = math.floor(lat * 1000000) / 1000000

    lon = "%.6f" % lon
    lat = "%.6f" % lat
    return lon, lat

def transformlat_lon_into_utm(x, y, zone=19):
    # verify the hemisphere
    # h_north = False
    # h_south = False
    # if (hemisphere == 'N'):
    #     h_north = True
    # elif (hemisphere == 'S'):
    #     h_south = True
    # else:
    #     print("Unknown hemisphere: " + hemisphere)

    proj_in = Proj(proj='utm', zone=zone, ellps='WGS84',preserve_units='m')

    lon, lat = proj_in(x, y)

    # just printing the floating point number with 6 decimal points will round it
    lon = math.floor(lon * 1000000) / 1000000
    lat = math.floor(lat * 1000000) / 1000000

    lon = "%.6f" % lon
    lat = "%.6f" % lat
    return lon, lat
# 32619
def trans2(x,y):
    transformer = Transformer.from_crs("epsg:32619", "epsg:4326")
    lat, lon = transformer.transform(x, y)
    return lat,lon

def get_epsg_from_wkt(wkt):
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(wkt)
    epsg = spatial_ref.GetAuthorityCode(None)
    return int(epsg)


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
    return coords[1],coords[0]


# img.GetProjection().split(' ')[5].split('"')[0][:-1]
if __name__ == '__main__':
    root = '../data/SPARCS/'
    temp_list = os.listdir(root)
    img_name_list = [x for x in temp_list if x.endswith('data.tif')]
    with open('lon_lat_srtm.txt', 'r+') as f:
        lon_lat_text = f.readlines()
    f.close()
    # srtm_name,top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = lon_lat_text[0].split(' ')
    for i in range(len(img_name_list)):
        img = gdal.Open(os.path.join(root, img_name_list[i]))
        top_left_corner_lon, top_left_corner_lat, bottom_right_corner_lon, bottom_right_corner_lat = project_xy(img)
        # top_left_corner_lon, top_left_corner_lat = geo2lonlat(img, top_left_corner_lon, top_left_corner_lat)
        # bottom_right_corner_lon, bottom_right_corner_lat = geo2lonlat(img,bottom_right_corner_lon, bottom_right_corner_lat)
        zone = img.GetProjection().split(' ')[5].split('"')[0][:-1]
        zone = int(zone)
        p1 = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units='m')
        top_left_corner_lon, top_left_corner_lat = p1(top_left_corner_lon, top_left_corner_lat, inverse=True)
        bottom_right_corner_lon, bottom_right_corner_lat = p1(bottom_right_corner_lon, bottom_right_corner_lat,inverse=True)
        for j in range(len(lon_lat_text)):
            srtm_name, top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = lon_lat_text[j].split(' ')
            top_left_srtm_lon, top_left_srtm_lat, bottom_right_srtm_lon, bottom_right_srtm_lat = \
            np.float64(top_left_srtm_lon),np.float64(top_left_srtm_lat),np.float64(bottom_right_srtm_lon),np.float64(bottom_right_srtm_lat)
            if (top_left_corner_lon>=top_left_srtm_lon) and (top_left_corner_lat<=top_left_srtm_lat):
                if (bottom_right_corner_lon<=bottom_right_srtm_lon) and (bottom_right_corner_lat>=bottom_right_srtm_lat):
                    with open('SPARCS_SRTM_name_list.txt', 'a+') as f:
                        f.writelines(img_name_list[i] + ' ' + srtm_name)
                        f.write('\n')
    f.close()