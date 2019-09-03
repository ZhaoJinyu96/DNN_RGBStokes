# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:50:09 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import modules.my_matplotlib_funcs as myplt
from modules.myFuncs import find_same_id
from modules.myFuncs import loadParamsFromConditiontxt
from modules.myNormalMapUtils import calcAngError, extractObjArea



# Load params
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["recError_per_sceneNum"]
    
data_names  = params["data_names"]
resultnames = params["resultnames"]
epochs = params["epochs"]
colors  = params["colors"]
maskFlag = params["maskFlag"]
saveFlag = params["saveFlag"]
savename = params["savename"]

# Make figure space
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ax.set_title("Reconstruction Error per Number of Images", fontsize=16)
ax.set_xlabel("Number of Images", fontsize=14)
ax.set_ylabel("Error [deg]", fontsize=14)


for color, data_name in zip(colors, data_names):
    
    imgNum, err_perImgNum = [], []
    
    for resultname, epoch in zip(resultnames, epochs):
        resultpath = "../normal_estimation/Result/{}/".format(resultname)
        
        imgNum.append(int(loadParamsFromConditiontxt(resultpath+"condition.txt")["trainingdatanum"]))
        
        gtfolder = "../normal_estimation/Picture/eval_dataset/{}/".format(data_name)
        gtnames = os.listdir(gtfolder+"/gt/")
        if maskFlag:
            masknames = os.listdir(gtfolder+"/mask/")
        
        recfolder = resultpath+"/Inference/epoch-{}/eval_dataset/{}/".format(epoch, data_name)
        recnames = os.listdir(recfolder)
        
        errs = np.empty(0, dtype=np.float32)
        
        for gtname in gtnames:
            recname = find_same_id(gtname, recnames)
            
            gt = cv2.imread(gtfolder+"/gt/"+gtname, -1).astype(np.float32)
            recon = cv2.imread(recfolder+recname, -1).astype(np.float32)
            
            # Calc error
            ang_deg = calcAngError(gt, recon)
            
            if maskFlag:
                maskname = find_same_id(gtname, masknames)
                mask = cv2.imread(gtfolder+"/mask/"+maskname, -1).astype(np.float32)
                ang_deg = extractObjArea(ang_deg, mask)
                
            mn, std = np.mean(ang_deg), np.std(ang_deg)
            errs = np.append(errs, ang_deg)
            
        err_perImgNum.append(np.mean(errs))
    
    # Plot
    ax.plot(imgNum, err_perImgNum,
            label=data_name, color=color, marker="o")
    
    ax.legend(bbox_to_anchor=(1.05, 1))
    
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
            
if saveFlag:
    myplt.my_savefig(fig, savename)
