# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:08:49 2018

@author: 0000145046
"""

import sys
sys.path.append("../")

from joblib import Parallel, delayed
import yaml
import numpy as np
import cv2

from mymodules.myutils.pathutils import find_all_csvpath_recursively
from mymodules.myutils.imageutils import MAX_16BIT


def makeMaskImagefromS0(img):
    mask_Black = (img[:,:,0] > 0)*(img[:,:,1] > 0)*(img[:,:,2] > 0)
    mask_White = (img[:,:,0] < MAX_16BIT)*(img[:,:,1] < MAX_16BIT)*(img[:,:,2] < MAX_16BIT)
    
    # Make mask image
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[:,:,0][mask_Black * mask_White] = 1
    mask[:,:,1][mask_Black * mask_White] = 1
    mask[:,:,2][mask_Black * mask_White] = 1
    
    mask *= 255
    return mask


def subprocess(csvpath, maskpath, imgnames):
    s0path = csvpath.joinpath("train/", imgnames[0])
    # Load gt image
    img = cv2.imread(str(s0path), -1).astype(np.float32)
    mask = makeMaskImagefromS0(img)

    cv2.imwrite(str(maskpath.joinpath(imgnames[3])), mask)


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["makeMaskImage"]
    
csvpaths = find_all_csvpath_recursively(params["csvpath"])

for csvpath in csvpaths:
    maskpath = csvpath.joinpath("mask")
    
    if maskpath.exists():
        pass
    else:
        maskpath.mkdir()
        
        csv = np.loadtxt(csvpath.joinpath("path.csv"), delimiter=",", dtype=str)
        Parallel(n_jobs=-1)([delayed(subprocess)(
                csvpath, maskpath, csv[i, :]) for i in range(csv.shape[0])])
