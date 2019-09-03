# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:11:14 2019

@author: 0000145046
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

import common_modules.myCnnModules.myNetwork as myNetwork

import common_modules.my_matplotlib_funcs as myplt
from common_modules.myFuncs import loadParamsFromConditiontxt

from chainer import serializers


#--------------------------
# Define parameter
#--------------------------
task         = "normal_estimation"
result       = "20190301_prim6th_s0DopPhase"
epoch_num    = 40
saveFlag     = False
nrows, ncols = (8, 8)
vmin, vmax   = (-1, -1) # if vmin==vmax, adaptive min and max will be assigned.





if task == "normal_estimation":
    filepath = "./normal_estimation/Result/" + result + "/"
    out_size = 3
elif task == "dif_spe_separation":
    filepath = "./dif_spe_separation/Result/" + result + "/"
    out_size = 18

# load condition.txt
params = loadParamsFromConditiontxt(filepath+"/condition.txt")
networkname = params["networkname"]
modalname   = params["modal"]

# define labels for figure
if modalname == "s0DopPhase":
    labels = ["s0_R",    "s0_G",    "s0_B",
              "DoP_R",   "DoP_G",   "DoP_B",
              "Phase_R", "Phase_G", "Phase_B"]
    
elif modalname == "only_s0":
    labels = ["s0_R", "s0_G", "s0_B"]
    
elif modalname == "only_s0_gray3ch":
    labels = ["s0_1st", "s0_2nd", "s0_3rd"]

# Load network and assign learned weights
net = myNetwork.networkSelect(networkname, out_size)
serializers.load_npz(filepath + "Model/snapshot_epoch-{:d}".format(epoch_num), net)


# Load 1st layer
if networkname == "PBR_CVPR2017_mod":
    conv0 = net.enc1.__getitem__(0).W.data
elif networkname == "ResNet34":
    conv0 = net.conv_init.W.data

out_ch, in_ch, w, h = conv0.shape
if nrows*ncols != out_ch:
    nrows, ncols = None, None
if vmin == vmax:
    vmin, vmax = (np.min(conv0), np.max(conv0))


# Plot
cmap = mpl.cm.Greys
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
for innum in range(in_ch):
    
    fig = plt.figure(figsize=(6,5))
    fig.suptitle(labels[innum], fontsize=18)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    
    axes = []
    for outnum in range(nrows*ncols):
        ax = plt.subplot(gs[outnum])
        img = ax.matshow(conv0[outnum,innum,:,:], cmap=cmap, norm=norm)
        myplt.hidden_all_axis(ax)
        axes.append(ax)
    
    cbar = fig.colorbar(img, ax=axes)
    
    if saveFlag:
        myplt.my_savefig(
                fig,
                filepath+"/Inference/epoch-{}/".format(epoch_num)+labels[innum]+".png")
    
    fig.show()
