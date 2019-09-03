# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:57:41 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

import mymodules.myutils.mplutils as myplt
from mymodules.myutils.pathutils import find_same_id
from mymodules.myutils.pathutils import find_same_id_s0
from mymodules.myutils.pathutils import returnSortedEpochNum
from mymodules.myutils.pathutils import  returnGTPaths
from mymodules.myutils.pathutils import  returnRecPaths
from mymodules.myutils.pathutils import  returnS0Paths
from mymodules.myutils.pathutils import  returnMaskPaths
from mymodules.myutils.imageutils import calcAngError, extractMaskArea


def plot_recErr_per_data(task, data_name, maskFlag, epochNums, resultname, color, ax):
    gtpaths = returnGTPaths(task, data_name)
    s0paths = returnS0Paths(task, data_name)
    
    if maskFlag:
            maskpaths = returnMaskPaths(task, data_name)
    
    err_perEpoch = []
    for epochNum in epochNums:
        recpaths = returnRecPaths(task, resultname, epochNum, data_name)
        
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

        err_perEpoch.append(np.mean(errs))
    
    ax.plot(epochNums,
            err_perEpoch, label=data_name.split("/")[1],
            color=color,
            marker=shape)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


# Main
# ----------------------
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["recError_per_data"]
    
task        = params["task"]
data_names1 = params["data_names1"]
data_names2 = params["data_names2"]
resultname  = params["resultname"]
shapes1     = params["shapes1"]
shapes2     = params["shapes2"]
colors      = params["colors"]
epochLim    = params["epochLim"]
maskFlag    = params["maskFlag"]
saveFlag    = params["saveFlag"]

# Input check
if len(data_names1) == len(shapes1) and\
    len(data_names2) == len(shapes2):
        pass
else:
    raise ValueError("Input numbers do not match.")
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Reconstruction Error per Epoch", fontsize=16)
ax.set_xlabel("epoch num", fontsize=12)
ax.set_ylabel("error [deg]", fontsize=12)
ax.set_xlim(0, epochLim)


# load learned result
result_epochs = [
    n.name for n in pathlib.Path("../{}/Result/{}/Model/".format(task, resultname)).iterdir()]
epochNums = returnSortedEpochNum(result_epochs)

for data_name, shape in zip(data_names1, shapes1):
    plot_recErr_per_data(
            task, data_name, maskFlag, epochNums, resultname, colors[0], ax)

for data_name, shape in zip(data_names2, shapes2):
    plot_recErr_per_data(
            task, data_name, maskFlag, epochNums, resultname, colors[1], ax)
    
if saveFlag:
    myplt.my_savefig(fig, "{}.png".format(saveFlag))
