# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:40:43 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import modules.myPolarFuncs as mpl
import modules.myDifSpeFuncs as mds

import os
import yaml
import numpy as np
import cv2


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["makeNormalFromDiffuse"]
    
result = params["result"]
epoch = params["epoch"]

path = "../dif_spe_separation/Result/{}/Inference/epoch-{}/test/".format(result, epoch)
dircs = os.listdir(path)

for dirc in dircs:        
    imNames = os.listdir(path+dirc)
    dopPhasePair = mds.makePairs_testDopPhase(imNames)
    
    savepath = path+dirc+"/normal"
    if not os.path.isdir(savepath):
            os.makedirs(savepath)
    
    for idname in dopPhasePair.keys():
        dop = cv2.imread("{}{}/{}".format(path,dirc,dopPhasePair[idname][0]), -1).astype(np.float32)
        phase = cv2.imread("{}{}/{}".format(path,dirc,dopPhasePair[idname][1]), -1).astype(np.float32)

        dop = dop / 65535.
        phase = phase * 180. / 65535.

        normal = mpl.calc_normal_from_dop_and_phase(dop, phase)
        
        cv2.imwrite("{}/{}.png".format(savepath, idname), normal.astype(np.uint8))

