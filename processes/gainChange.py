# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:48:07 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import pathlib
import shutil
import yaml

import numpy as np

from mymodules.myutils.imageutils import MAX_16BIT
from mymodules.myclass.augmentations import mydataAugmentation_difSpe
import mymodules.myutils.polarutils as mpl
import mymodules.myutils.pathutils as ptu


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["gainChange"]
    
csvpaths = ptu.find_all_csvpath_recursively(params["path"])
gains = params["gains"]

for gain in gains:
    data = mydataAugmentation_difSpe.DataAugmentation_difSpe_Base(
            csvpaths, 16, "gain_{}".format(str(gain)))
    
    for path in data.path_list:
        data.make_savedir(path)
        savepath = pathlib.Path(str(path) + "_" + data.operationName)
        
        csv = np.loadtxt(
                        path.joinpath("path.csv"), delimiter=",", dtype=str)
    
        for i in range(csv.shape[0]):
            s0, s1, s2, gt, s0spe, s1spe, s2spe, s0dif, s1dif, s2dif = data.load_image(path, csv[i,:])
            
            i0, i45, i90, i135 = mpl.calc_fourPolar_from_stokes(s0, s1, s2)
            i0_spe, i45_spe, i90_spe, i135_spe = mpl.calc_fourPolar_from_stokes(s0spe, s1spe, s2spe)
            i0_dif, i45_dif, i90_dif, i135_dif = mpl.calc_fourPolar_from_stokes(s0dif, s1dif, s2dif)
            
            # multiply gain
            i0 *= gain
            i45 *= gain
            i90 *= gain
            i135 *= gain
            
            i0_spe *= gain
            i45_spe *= gain
            i90_spe *= gain
            i135_spe *= gain

            i0_dif *= gain
            i45_dif *= gain
            i90_dif *= gain
            i135_dif *= gain
            
            # return back to s0s1s2
            s0, s1, s2 = mpl.calc_s0s1s2_from_fourPolar(i0, i45, i90, i135)
            s0spe, s1spe, s2spe = mpl.calc_s0s1s2_from_fourPolar(i0_spe, i45_spe, i90_spe, i135_spe)
            s0dif, s1dif, s2dif = mpl.calc_s0s1s2_from_fourPolar(i0_dif, i45_dif, i90_dif, i135_dif)
            
            # de-normalize
            s0, s1, s2 = mpl.de_normalize_s0s1s2(s0, s1, s2, MAX_16BIT)
            s0spe, s1spe, s2spe = mpl.de_normalize_s0s1s2(s0spe, s1spe, s2spe, MAX_16BIT)
            s0dif, s1dif, s2dif = mpl.de_normalize_s0s1s2(s0dif, s1dif, s2dif,MAX_16BIT)
            
            s0 = np.clip(s0, 0, MAX_16BIT)
            s1 = np.clip(s1, 0, MAX_16BIT)
            s2 = np.clip(s2, 0, MAX_16BIT)
            
            s0spe = np.clip(s0spe, 0, MAX_16BIT)
            s1spe = np.clip(s1spe, 0, MAX_16BIT)
            s2spe = np.clip(s2spe, 0, MAX_16BIT)
            
            s0dif = np.clip(s0dif, 0, MAX_16BIT)
            s1dif = np.clip(s1dif, 0, MAX_16BIT)
            s2dif = np.clip(s2dif, 0, MAX_16BIT)
            
            data.save_image(savepath, csv[i,:],
                (s0, s1, s2, gt, s0spe, s1spe, s2spe, s0dif, s1dif, s2dif))
            
        # copy other files
        shutil.copy(str(path.joinpath("path.csv")), str(savepath.joinpath("path.csv")))
            
            