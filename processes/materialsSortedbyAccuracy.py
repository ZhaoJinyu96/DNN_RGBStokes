# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:04:01 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import pprint

import cv2
import numpy as np

from mymodules.myutils.pathutils import find_same_id
from mymodules.myutils.pathutils import find_same_id_s0
from mymodules.myutils.pathutils import  returnGTPaths
from mymodules.myutils.pathutils import  returnRecPaths
from mymodules.myutils.pathutils import  returnS0Paths
from mymodules.myutils.pathutils import  returnMaskPaths
from mymodules.myutils.imageutils import calcAngError, extractMaskArea


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["materialsSortedbyAccuracy"]
    
task        = params["task"]
data_names  = params["data_names"]
resultname = params["resultname"]
epochnum    = params["epochnum"]
maskFlag    = params["maskFlag"]

resultdic = {}
for data_name in data_names:
    gtpaths = returnGTPaths(task, data_name)
    s0paths = returnS0Paths(task, data_name)
    recpaths = returnRecPaths(task, resultname, epochnum, data_name)
    if maskFlag:
            maskpaths = returnMaskPaths(task, data_name)
            
    errs = np.empty(0, dtype=np.float32)
    for gtpath in gtpaths:
        s0path  = find_same_id_s0(gtpath, s0paths)
        recpath = find_same_id(gtpath, recpaths)
        
        gt  = cv2.imread(str(gtpath), -1).astype(np.float32)
        s0  = cv2.imread(str(s0path), -1).astype(np.float32)
        rec = cv2.imread(str(recpath), -1).astype(np.float32)
        
        # Calc masks
        noObj_mask = (np.sum(gt, axis=2) > 0)
        s0_mask    = (np.sum(s0/3, axis=2) > params["s0threshold"])
        
        # Calc error
        ang_deg = calcAngError(gt, rec)
        
        # consider mask area
        if maskFlag:
            maskpath = find_same_id(gtpath, maskpaths)
            mask = cv2.imread(str(maskpath), -1).astype(np.float32)
            
            ang_deg    = extractMaskArea(ang_deg, mask, maskFlag)
            noObj_mask = extractMaskArea(noObj_mask, mask, maskFlag)
            s0_mask    = extractMaskArea(s0_mask, mask, maskFlag)
            
        ang_deg = ang_deg[noObj_mask * s0_mask]
        errs = np.append(errs, ang_deg)
    
    resultdic[data_name] = np.mean(errs)

pprint.pprint(sorted(resultdic.items(), key=lambda x:x[1]))