# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:40:25 2019

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
from mymodules.myutils.pathutils import  returnGTPaths
from mymodules.myutils.pathutils import  returnRecPaths
from mymodules.myutils.pathutils import  returnS0Paths
import mymodules.myutils.imageutils as nm
from mymodules.myutils.imageutils import MAX_16BIT
import mymodules.myutils.mplutils as myplt


# load Params
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
    raise ValueError("maskFlag shoud be False")

# main process
#------------------------
x, y = [], [] # for plot scatter graph
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
    #calc gt zenith
    zenith_deg = nm.calcZenithDegMap(gt[:,:,::-1])
    # calc depth error
    depth_err = nm.normalErr2depthErr(zenith_deg, ang_deg)
    
    # prepare figure
    fig = plt.figure(figsize=(9, 4.5))
    gs = gridspec.GridSpec(
            3, 3, height_ratios=[1,1,1], width_ratios=[4,1,1])   
    ax_main = plt.subplot(gs[:, 0])
    
    # prepare cmap
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    
    # show depth error
    img = ax_main.matshow(depth_err, cmap=cmap, norm=norm)
    ax_main.set_title("Depth error [mm]", fontsize=14)
    myplt.hidden_all_axis(ax_main)
    
    # add cmap
    cmapdivider = make_axes_locatable(ax_main)
    cax = cmapdivider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    
    # extract depth error by zenith angle
    zenith_step = 15
    cnt = 0
    for zenith_num in range(0, 90, zenith_step):
        
        zen_min = zenith_num
        zen_max = zenith_num+zenith_step
        
        zenith_mask = (zenith_deg >= zen_min) * (zenith_deg < zen_max)
        
        # polt area
        ax = plt.subplot(gs[cnt%3, int(cnt/3)+1])
        ax.matshow((zenith_mask*noObj_mask).astype(np.uint8))
        ax.set_title(r"{} $\leq$ $\theta$ $<$ {}".format(zen_min, zen_max),
                     pad=0.008, fontsize=8)
        myplt.hidden_all_axis(ax)
        
        # calc average error of each area
        mask_conv = zenith_mask*noObj_mask*s0_Darkmask*s0_Whitemask
        
        if np.sum(mask_conv) > 0:
            
            depth_angMasked = depth_err[mask_conv]
            err_mean = np.mean(depth_angMasked)
            
            myplt.drawtextInAxesBttomLeft(
                    fig, ax,
                    "mean:{:>5.2f} [mm]".format(err_mean),
                    -0.008, 8)
            
            x.append(zen_max), y.append(err_mean)
            
        else:
            err_mean = 0
            myplt.drawtextInAxesBttomRight(
                    fig, ax, "No area", -0.008, 8)
        
        cnt += 1
    
    if saveFlag:
        myplt.my_savefig(
                fig,
                str(savefile.joinpath("{}.png".format(extractID(gtpath.name))))
                )
        
    fig.show()


# scatter err by zenith angle
#---------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, y)

for zenith_num in range(0, 90, zenith_step):
    y_ex = np.array(y)[np.array(x)==zenith_num+zenith_step]
    mn = np.mean(y_ex)
    ax.text(
            zenith_num+zenith_step, 30,
            "{:.2f}".format(mn),
            ha = "center")

ax.set_title("Depth error of images", fontsize=16)
ax.set_xlabel("Zenith angle [deg]", fontsize=12)
ax.set_ylabel("Depth error [mm]", fontsize=12)
ax.set_yscale("log")
ax.grid(which="both")

if saveFlag:
        myplt.my_savefig(fig,
                         str(savefile.joinpath("scatter.png"))
                         )
        
fig.show()

