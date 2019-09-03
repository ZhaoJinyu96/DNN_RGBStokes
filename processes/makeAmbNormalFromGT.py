# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:58:29 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import pathlib
import yaml

import numpy as np
import cv2

from mymodules.myutils.imageutils import makeAmbNormal




with open("./parameters_process.yml") as f:
    params = yaml.load(f)["makeAmbNormalFromGT"]

dircs = params["path"]

for dirc in dircs:
    dirc = pathlib.Path(dirc)
    gtNames = dirc.joinpath("gt").glob("*.png")
    
    savepath = dirc.joinpath("gt_amb")
    savepath.mkdir()
    
    for gtName in gtNames:
        gtImg = cv2.imread(str(gtName), -1).astype(np.float32)
        ambImg = makeAmbNormal(gtImg[:,:,::-1])
        
        cv2.imwrite(
                str(savepath.joinpath(gtName.name)),
                ambImg[:,:,::-1].astype(np.uint16))
        
    

