# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:45:09 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import pathlib

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mymodules.myutils.pathutils import find_same_id
from mymodules.myutils.pathutils import find_same_id_s0
from mymodules.myutils.pathutils import extractID
from mymodules.myutils.pathutils import returnGTPaths
from mymodules.myutils.pathutils import returnRecPaths
from mymodules.myutils.pathutils import returnS0Paths
from mymodules.myutils.pathutils import returnMaskPaths
import mymodules.myutils.imageutils as nm
from mymodules.myutils.imageutils import MAX_16BIT
import mymodules.myutils.mplutils as myplt


# load params
#------------------------
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["recError_per_image"]
    
task       = params["task"]
gtfolder   = params["gtfolder"]
resultname = params["resultname"]
epochnum   = params["epochnum"]
maskFlag   = params["maskFlag"]
saveFlag   = params["saveFlag"]

gtpaths = returnGTPaths(task, gtfolder)
s0paths = returnS0Paths(task, gtfolder)
recpaths = returnRecPaths(task, resultname, epochnum, gtfolder)


if saveFlag:
    savefile = pathlib.Path(
            "../{}/Result/{}/Inference/epoch-{}/{}/{}".format(
                    task, resultname, epochnum, gtfolder, saveFlag))
    savefile.mkdir()

if maskFlag:
    maskfile = pathlib.Path(
            "../{}/Picture/{}/mask/".format(task, gtfolder))
    maskPaths = returnMaskPaths(task, gtfolder)


# main process
#------------------------
errs = np.empty(0, dtype=np.float32)

for gtpath in gtpaths:
    recpath = find_same_id(gtpath, recpaths)
    s0path  = find_same_id_s0(gtpath, s0paths) 

    # load images
    gt    = cv2.imread(str(gtpath), -1).astype(np.float32)
    recon = cv2.imread(str(recpath), -1).astype(np.float32)
    s0    = cv2.imread(str(s0path), -1).astype(np.float32)
    
    # mask pixels which do not have any values
    noObj_mask = (np.sum(gt, axis=2) > 0)
    # mask pixels whose values are smaller than threshold
    s0_Darkmask  = (np.sum(s0/3, axis=2) > params["s0threshold"])
    s0_Whitemask = (s0[:,:,0]<MAX_16BIT) + (s0[:,:,1]<MAX_16BIT) + (s0[:,:,2]<MAX_16BIT)
    
    # calc error
    ang_deg = nm.calcAngError(gt, recon)
    
    
    # plot
    fig = plt.figure()
    gs = gridspec.GridSpec(1,2,width_ratios=[1,2])
    
    # set axes
    ax0, ax1 = plt.subplot(gs[0]), plt.subplot(gs[1])
    
    # show normal map
    normdivider = make_axes_locatable(ax0)
    ax0_ = normdivider.append_axes("bottom", size="100%")
    ax0.imshow(gt[:,:,::-1]/65535.)
    ax0_.imshow(recon[:,:,::-1]/255.)
    fig.canvas.draw() # update the params of axes
    myplt.drawtextInAxesBttomLeft(fig, ax0, "GT normal", ypos=0.01)
    myplt.drawtextInAxesBttomLeft(fig, ax0_, "Recon normal", ypos=0.01)
    myplt.hidden_all_axis(ax0)
    myplt.hidden_all_axis(ax0_)
    
    # show reconstruction error
    # set colormap
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=60)
    # plot
    img = ax1.matshow(ang_deg, cmap=cmap, norm=norm)
    myplt.hidden_all_axis(ax1)
    ax1.set_title("Reconstruction Error", fontsize=14)
    # add colormap
    cmapdivider = make_axes_locatable(ax1)
    cax = cmapdivider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Angle [deg]", fontsize=14)
    
    # calc. mean and std of error
    if maskFlag:
        maskpath = find_same_id(gtpath, maskPaths)
        mask = cv2.imread(str(maskpath), -1).astype(np.float32)
        
        # apply mask
        ang_deg      = nm.extractMaskArea(ang_deg, mask, maskFlag)
        noObj_mask   = nm.extractMaskArea(noObj_mask, mask, maskFlag)
        s0_Darkmask  = nm.extractMaskArea(s0_Darkmask, mask, maskFlag)
        s0_Whitemask = nm.extractMaskArea(s0_Whitemask, mask, maskFlag)
    
    ang_deg = ang_deg[noObj_mask * s0_Darkmask * s0_Whitemask]
    mn, std = np.mean(ang_deg), np.std(ang_deg)
    errs = np.append(errs, ang_deg)
    
    fig.canvas.draw()
    myplt.drawtextInAxesBttomRight(fig, ax1,
                                   "mean:{:>8.2f} [deg]".format(mn), -0.04)
    myplt.drawtextInAxesBttomRight(fig, ax1,
                                   "std    :{:>8.2f} [deg]".format(std), -0.08)
    
    # save
    if saveFlag:
        myplt.my_savefig(fig,
                         str(savefile.joinpath("{}.png".format(extractID(gtpath.name))))
                         )
    
    fig.show()
    
print("mean is: {:.2f}".format(np.mean(errs)))
print("stdev is: {:.2f}".format(np.std(errs)))

    