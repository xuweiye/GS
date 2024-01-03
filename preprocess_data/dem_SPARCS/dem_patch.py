import os
from osgeo import gdal
import numpy as np
from preprocess_data.dem_SPARCS.dem_process import project_xy
from pyproj import Proj
from scipy import ndimage
gdal.PushErrorHandler('CPLQuietErrorHandler')
def geo2imagexy2(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    p1 = Proj(proj='utm', zone=19, ellps='WGS84', preserve_units='m')
    c,d = p1(trans[0],trans[3])
    a = np.array([[90.0, trans[2]], [trans[4], -90.0]])
    b = np.array([x - c, y - d])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

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
    dataset_path = '../../data/SPARCS'
    srtm_path = '../../data/SRTM_90米_DEM'
    with open('SPARCS_SRTM_name_list.txt', 'r+')as f:
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
        # top_left_corner_lon, top_left_corner_lat = geo2lonlat(img, top_left_corner_lon, top_left_corner_lat)
        # bottom_right_corner_lon, bottom_right_corner_lat = geo2lonlat(img, bottom_right_corner_lon,bottom_right_corner_lat)
        srtm_band = srtm.GetRasterBand(1).ReadAsArray()
        x1,y1 = geo2imagexy(srtm,top_left_corner_lon, top_left_corner_lat,x1_geo,y1_geo)
        x2, y2 = geo2imagexy(srtm, bottom_right_corner_lon, bottom_right_corner_lat, x1_geo, y1_geo)
        # x1,y1 = int((y0 - top_left_corner_lat)/dy),int((top_left_corner_lon - x0)/dx)
        # x2, y2 = int((y0 - bottom_right_corner_lat) / dy), int((bottom_right_corner_lon - x0) / dx)
        # x1,y1 = geo2imagexy2(srtm,top_left_corner_lon,top_left_corner_lat)
        # x2,y2 = geo2imagexy2(srtm,bottom_right_corner_lon,bottom_right_corner_lat)
        # print(srtm_band[int(x1):int(x2),int(y1):int(y2)].shape)
        output = srtm_band[int(x1):int(x2),int(y1):int(y2)]
        # output = torch.from_numpy(output)
        # up = nn.UpsamplingBilinear2d(size=(1000,1000))
        # optput = up(output)
        output_file = '../data/SPARCS_dem/' + dataset_name_list[i][:-8] + 'dem.tif'
        output = ndimage.zoom(output,(1000/output.shape[0],1000/output.shape[1]),order=3,mode='nearest')
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_file,1000,1000,1,gdal.GDT_UInt16)
        projection = img.GetProjection()
        dataset.SetProjection(projection)
        geotransform = img.GetGeoTransform()
        dataset.SetGeoTransform(geotransform)
        band = dataset.GetRasterBand(1)
        band.WriteArray(output)
        del dataset

