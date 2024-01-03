from osgeo import gdal
import os
from tqdm import tqdm
from xml.dom import minidom


if __name__ == '__main__':
    root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\mss'
    mask_root = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\mask_1'
    save_dir = r'D:\同步文件\BaiduSyncdisk\project_plus\my_net\data\WFV\raw\mask'
    root_list = os.listdir(root)
    mss_list = [x for x in root_list if x.endswith('tiff')]
    xml_list = [x[:-5] + '.xml' for x in mss_list]
    mask_list = mss_list
    for i in tqdm(range(len(mss_list))):
        if mss_list[i][:-5] == mask_list[i][:-5]:
            mss = gdal.Open(os.path.join(mask_root,mask_list[i]))
            # mss = gdal.Open(os.path.join(root,mss_list[i]))
            # xml_doc = minidom.parse(os.path.join(root,xml_list[i]))
            msstProj = mss.GetProjection()
            referencefileProj = mss.GetProjection()
            referencefileTrans = list(mss.GetGeoTransform())
            bandreferencefile = mss.GetRasterBand(1)
            x = 1500
            y = 1500
            referencefileTrans[1] *= 1
            referencefileTrans[5] *= 1
            nbands = 1
            output_image_path = os.path.join(save_dir, mss_list[i])
            driver = gdal.GetDriverByName('GTiff')
            output = driver.Create(output_image_path, x, y, nbands, bandreferencefile.DataType)
            output.SetGeoTransform(referencefileTrans)
            output.SetProjection(referencefileProj)
            data = mss.ReadRaster(
                buf_xsize=x, buf_ysize=y)
            output.WriteRaster(0, 0, x, y, data)
            output.FlushCache()
            for i in range(1):
                output.GetRasterBand(i + 1).ComputeStatistics(False)
            output.BuildOverviews('average', [2, 4, 8, 16])

        # xml_doc = minidom.parse(os.path.join(root,xml_list[i]))
        # sample_size = int(xml_doc.getElementsByTagName('ImageGSD')[0].firstChild.data)
        # print(sample_size)
