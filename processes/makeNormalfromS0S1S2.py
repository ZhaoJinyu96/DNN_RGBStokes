# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:20:10 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import pathlib

import numpy as np
import cv2

import mymodules.myutils.polarutils as mpl
from mymodules.myutils.imageutils import MAX_16BIT
from mymodules.myutils.pathutils import extractID


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["makeNormalfromS0S1S2"]

path = pathlib.Path(params["path"])

outPath = path.joinpath("modelNormal_plain")
outPath.mkdir()

# The following line ignores the difference between s0 and S0.
imgNames = list(path.joinpath("train").glob("*S0.png"))

for imgName in imgNames:
    s0_name = str(imgName)
    
    if s0_name.find("S0") == -1:
        raise ValueError("S0 should be a capital letter.")
    if s0_name.find("ID-") == -1:
        raise ValueError("Input image names should be started with 'ID-'")
    
    
    s1_name = str(imgName).replace("S0", "S1")
    s2_name = str(imgName).replace("S0", "S2")
    
    # load
    s0 = cv2.imread(s0_name, -1).astype(np.float32)
    s1 = cv2.imread(s1_name, -1).astype(np.float32)
    s2 = cv2.imread(s2_name, -1).astype(np.float32)
    
    s0_norm, s1_norm, s2_norm = mpl.normalize_s0s1s2(s0, s1, s2, MAX_16BIT)
    
    dop   = mpl.calc_dop_from_stokes(s0_norm, s1_norm, s2_norm) # (0, 1)
    phase = mpl.calc_polar_phase_from_stokes(s1_norm, s2_norm) # Degree [0, 180)
    
    normal = mpl.calc_normal_from_dop_and_phase(dop[:,:,1], phase[:,:,1])
    
    cv2.imwrite(
            str(outPath.joinpath("{}.png".format(extractID(imgName.name)))),
            normal.astype(np.uint8))
