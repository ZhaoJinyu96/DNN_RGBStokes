# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:51:57 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import os
import cv2
import numpy as np

from modules.myFuncs import find_all_csvpaths_recursively

MAX_16BIT = 65535


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["removeSaturatedImages"]
    
csvpaths = find_all_csvpaths_recursively(params["path"])

for csvpath in csvpaths:
    os.rename(csvpath+"path.csv", csvpath+"path_all.csv")
    csv = np.loadtxt(csvpath+"path_all.csv", delimiter=",", dtype=str)
    
    csv_dst = np.empty((0, 4), dtype=str)
    
    for imgnum in range(csv.shape[0]):
        # load s0
        img = cv2.imread(csvpath+"/train/"+csv[imgnum, 0], -1).astype(np.float32)
        
        # check whether there are saturated pixels or not
        mask_B = (img[:,:,0]==MAX_16BIT)
        mask_G = (img[:,:,1]==MAX_16BIT)
        mask_R = (img[:,:,2]==MAX_16BIT)
        
        mask_all = (mask_B + mask_G + mask_R)
        
        if np.cumsum(mask_all)[-1] > 0: # saturated area exists
            pass
        else:
            csv_dst = np.append(csv_dst, np.array([csv[imgnum,:]]), axis=0)
        
    np.savetxt(csvpath + "path.csv",
               csv_dst,
               delimiter=",",
               fmt="%s")
