# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:59:46 2019

@author: 0000145046
"""


import sys
sys.path.append("../")

import yaml
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

from modules.myFuncs import find_same_id, find_same_id_s0, returnFolderAndImNames_GtRecS0
import modules.myNormalMapUtils as nm
import modules.my_matplotlib_funcs as myplt


# load Params
#------------------------
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["recError_per_image"]
    
gtfolder,gtnames,recfolder,recnames,s0folder,s0names = returnFolderAndImNames_GtRecS0(params)

maskFlag = params["maskFlag"]
saveFlag = params["saveFlag"]

if saveFlag:
    savefile = "{}/{}/".format(recfolder, saveFlag)
    os.makedirs(savefile)

if maskFlag:
    maskfolder = "../{}/Picture/{}/mask/".format(params["task"], params["gtfolder"])
    masknames = os.listdir(maskfolder)

# main
#--------------------
for i, gtname in enumerate(gtnames):
    recname = find_same_id(gtname, recnames)
    s0name  = find_same_id_s0(gtname, s0names) 
    
    # load images
    gt    = cv2.imread(gtfolder+gtname, -1).astype(np.float32)
    recon = cv2.imread(recfolder+recname, -1).astype(np.float32)
    s0    = cv2.imread(s0folder+s0name, -1).astype(np.float32)
    
    # mask pixels which do not have any values
    noObj_mask = (np.sum(gt, axis=2) > 0)
    # mask pixels whose values are smaller than threshold
    s0_mask    = (np.sum(s0/3, axis=2) > params["s0threshold"])
    
    # calc error
    ang_deg = nm.calcAngError(gt, recon)
    
    #calc gt zenith
    zenith_deg = nm.calcZenithDegMap(gt[:,:,::-1])
    zenith_mask = (zenith_deg < 75)
    
    mask_all = noObj_mask * s0_mask * zenith_mask
    
    
    # plot
    #--------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("FoV [deg]", fontsize=12)
    ax.set_ylabel("The number of pixels", fontsize=12)
    ax.set_yscale("log")
    ax.grid(which="both")
    
    for dErr in range(1, 8):
        pixNum = []
        for deg in range(0,180):
            pixNum.append(nm.calcPixNfromFoVDepthErr(
                    deg, zenith_deg, ang_deg, mask_all, depth_err=dErr))
        

        ax.plot(pixNum, label=r"$\Delta$d={}".format(dErr))
    
    ax.legend()
        
    if saveFlag:
        myplt.my_savefig(fig, savefile+"pixnum{}.png".format(i))

    fig.show()

