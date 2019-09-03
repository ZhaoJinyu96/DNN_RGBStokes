# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:18:12 2019

@author: 0000145046
"""


import sys
sys.path.append("../")

import yaml
import os
import numpy as np
import cv2

import modules.myFuncs as mf
from modules.myPolarFuncs import calc_PolarImg_fromS0DopPhase
import modules.my_matplotlib_funcs as myplt

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# load params
#------------------------
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["recError_per_image"]

if params["task"] == "normal_estimation":
    raise ValueError("This visualizer is only allowed to dif_spe_separation.")
    

gtfolder  = "../{}/Picture/{}/gt/".format(
        params["task"], params["gtfolder"])
gtnames = os.listdir(gtfolder)

mixfolder = "../{}/Picture/{}/train/".format(
        params["task"], params["gtfolder"])
mixnames = os.listdir(mixfolder)

diffolder = "../{}/Picture/{}/gt_difspe/{}_dif/train/".format(
        params["task"], params["gtfolder"], mf.calcDesignatedPathBlock(gtfolder, 2, True))
difnames = os.listdir(diffolder)

recfolder = "../{}/Result/{}/Inference/epoch-{}/{}/".format(
        params["task"], params["resultname"], params["epochnum"], params["gtfolder"])
recnames = os.listdir(recfolder)

maskFlag = params["maskFlag"]
saveFlag = params["saveFlag"]

if saveFlag:
    savefile = "{}/{}/".format(recfolder, saveFlag)
    os.makedirs(savefile)
    
if maskFlag:
    maskfolders = "../{}/Picture/{}/mask/".format(params["task"], params["gtfolder"])
    masknames = os.listdir(maskfolders)
    

# main process
#------------------------
for i, gtname in enumerate(gtnames):
    gtname_s0 = mf.find_same_id_s0(gtname, difnames)
    gtname_s1 = mf.find_same_id_s1(gtname, difnames)
    gtname_s2 = mf.find_same_id_s2(gtname, difnames)
    
    mixname_s0 = mf.find_same_id_s0(gtname, mixnames)
    
    recname_s0Dif = mf.find_same_id_fromKey(gtname, recnames, "s0Dif")
    recname_DopDif = mf.find_same_id_fromKey(gtname, recnames, "DopDif")
    recname_PhaseDif = mf.find_same_id_fromKey(gtname, recnames, "PhaseDif")
    
    # load images
    #----------------------------
    gtimg_s0 = cv2.imread(diffolder+gtname_s0, -1).astype(np.float32)
    gtimg_s0 = np.sum(gtimg_s0, axis=2) / 3.
    
    gtimg_s1 = cv2.imread(diffolder+gtname_s1, -1).astype(np.float32)
    gtimg_s1 = np.sum(gtimg_s1, axis=2) / 3.
    gtimg_s1 = gtimg_s1 * 2 - 65535
    
    gtimg_s2 = cv2.imread(diffolder+gtname_s2, -1).astype(np.float32)
    gtimg_s2 = np.sum(gtimg_s2, axis=2) / 3.
    gtimg_s2 = gtimg_s2 * 2 - 65535
    
    miximg_s0 = cv2.imread(mixfolder+mixname_s0, -1).astype(np.float32)
    
    recimg_s0Dif = cv2.imread(recfolder+recname_s0Dif, -1).astype(np.float32)
    recimg_DopDif = cv2.imread(recfolder+recname_DopDif, -1).astype(np.float32)
    recimg_PhaseDif = cv2.imread(recfolder+recname_PhaseDif, -1).astype(np.float32)
    
    # calc four polar images
    #----------------------------
    gt_i0   = (gtimg_s0 + gtimg_s1) / 2.
    gt_i45  = (gtimg_s0 + gtimg_s2) / 2.
    gt_i90  = (gtimg_s0 - gtimg_s1) / 2.
    gt_i135 = (gtimg_s0 - gtimg_s2) / 2.
    
    rec_i0  = calc_PolarImg_fromS0DopPhase(
            recimg_s0Dif, recimg_DopDif, recimg_PhaseDif,
            0)
    rec_i45 = calc_PolarImg_fromS0DopPhase(
            recimg_s0Dif, recimg_DopDif, recimg_PhaseDif,
            45)
    rec_i90 = calc_PolarImg_fromS0DopPhase(
            recimg_s0Dif, recimg_DopDif, recimg_PhaseDif,
            90)
    rec_i135 = calc_PolarImg_fromS0DopPhase(
            recimg_s0Dif, recimg_DopDif, recimg_PhaseDif,
            135)
    
    err_i0   = np.abs(gt_i0 - rec_i0)
    err_i45  = np.abs(gt_i45 - rec_i45)
    err_i90  = np.abs(gt_i90 - rec_i90)
    err_i135 = np.abs(gt_i135 - rec_i135)
    
    err_all = (err_i0 + err_i45 + err_i90 + err_i135) / 4.
    
    # plot
    #-----------------------------
    fig = plt.figure()
    gs = gridspec.GridSpec(
            1, 2, width_ratios=[1,2.5])
    
    ax0 = plt.subplot(gs[:, 0])
    ax = plt.subplot(gs[:, 1])
    
    # plot source s0
    ax0.imshow(miximg_s0[:,:,::-1] / 65535.)
    myplt.hidden_all_axis(ax0)
    
    # cmap
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=1000)
    # plot
    img = ax.matshow(err_all,
                     cmap=cmap, norm=norm)
    myplt.hidden_all_axis(ax)
    # add cmap
    cmapdivider = make_axes_locatable(ax)
    cax = cmapdivider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Error of pixel value", fontsize=14)
    
    # add mean and std
    mn, std = np.mean(err_all), np.std(err_all)
    myplt.drawtextInAxesBttomRight(fig, ax,
                                   "mean:{:>8.2f}".format(mn), -0.0)
    myplt.drawtextInAxesBttomRight(fig, ax,
                                   "std    :{:>8.2f}".format(std), -0.04)
    
    # save
    if saveFlag:
        myplt.my_savefig(fig, savefile+"error{:03}.png".format(i))
    
    fig.show()
    
