import cv2 as cv
import torch
import random

import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datacube, mask):
        #assert img.size == label.size
        #print img.shape
        for t in self.transforms:
            #print t
            datacube, mask = t(datacube, mask)
        #print img.shape

        return datacube, mask


####joint-transform
class Resize(object):
    def __init__(self,r_size,interpolation=cv.INTER_NEAREST):
        size_h = r_size[1]
        size_w = r_size[0]
        self.r_size = r_size #input is a tuple. size: (w,h) like opencv
        self.interpolation = interpolation

    def __call__(self,datacube):
        datacube_result = cv.resize(datacube,dsize=self.r_size,interpolation=self.interpolation)
        return datacube_result

class RandomImgCrop(object):
    def __init__(self,crop_size):
        self.crop_size = crop_size

    def __call__(self,datacube,mask):
        #img format: channels, h, w
        #label format: 1,h,w

        h = mask.shape[1]
        w = mask.shape[2]
        # print h
        # print w

        limit = min(h,w)
        if self.crop_size >limit:
            self.crop_size = limit

        #lefttop
        h_range = h - self.crop_size
        w_range = w - self.crop_size

        h_r = random.random()
#       print h_r
        w_r = random.random()
#       print w_r

        lt_h = max(int(h_r * h_range)-1,0)
        lt_w = max(int(w_r * w_range)-1,0)
        datacube_crop = datacube[:,lt_h:lt_h+self.crop_size,lt_w:lt_w+self.crop_size]
        #print img
        mask_crop = mask[:,lt_h:lt_h+self.crop_size,lt_w:lt_w+self.crop_size]
        return datacube_crop,mask_crop

class RotateTransform(object):
    def __init__(self):
        pass

    def __call__(self,datacube,mask):
        selectedOperation = int(random.random()*4)
        datacube = torch.rot90(datacube,k=selectedOperation,dims=[1,2])
        mask = torch.rot90(mask,k=selectedOperation,dims=[1,2])
        return datacube,mask

class RandomHorizontallyFlip(object):
    def __call__(self, sample):
        img,mask = sample
        img,mask = torch.from_numpy(img),torch.from_numpy(mask)
        if random.random() < 0.5:
            return torch.flip(img,[1,2]).numpy(),torch.flip(mask,[1,2]).numpy()
        return img.numpy(),mask.numpy()