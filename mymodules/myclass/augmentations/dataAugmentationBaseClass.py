# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:12:39 2018

@author: 0000145046
"""

# add path
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.joinpath("../../")))

import random
from joblib import Parallel, delayed
import numpy as np

from mymodules.myutils.imageutils import MAX_8BIT, MAX_16BIT

    

class DataAugmentationBaseClass:
    """
    base class of data augmentation object.
    If you want to implement the new augmentation class, 
    please inherit this class, and newly implement "operation" function.
    """
    def __init__(self, path_list, bit, operationName):

        self.path_list     = path_list
        self.operationName = operationName
        self.bit           = None
        self.max           = None
        
        if bit == 8:
            self.bit = np.uint8
            self.max = MAX_8BIT
        elif bit == 16:
            self.bit = np.uint16
            self.max = MAX_16BIT

      
    def run_augmentation_and_save(self):
        for path in self.path_list:
            if "flip_horizontal" in str(path) or \
                          "blur" in str(path) or \
                         "noise" in str(path):
                    pass
                
            else:
                self.make_savedir(path)
                savepath = pathlib.Path(str(path) + "_" + self.operationName)
                
                csv = np.loadtxt(
                        path.joinpath("path.csv"), delimiter=",", dtype=str)
                
                processedImageList = Parallel(n_jobs=-1)([delayed(self.subprocess)(
                        path, savepath, csv[i,:]) for i in range(csv.shape[0])])
    
                csv_dst = np.empty((0, 4), dtype=str)
                for processedImage in processedImageList:
                    if type(processedImage) == np.ndarray:
                        csv_dst = np.append(
                                csv_dst, np.array([processedImage]),
                                axis=0)

                np.savetxt(savepath.joinpath("path.csv"),
                           csv_dst,
                           delimiter=",",
                           fmt="%s")
                
    
    def subprocess(self, path, savepath, imgnames):
        if random.randint(0,1):
            imgs = self.load_image(path, imgnames)
            imgs = self.augmentation(imgs)
            
            savepath = pathlib.Path(str(path) + "_" + self.operationName)
            self.save_image(savepath, imgnames, imgs)
            
            return imgnames
        else:
            return 0

        
    def make_savedir(self, path):
        raise NotImplementedError
        
    def load_image(self, path, names):
        raise NotImplementedError
    
    def save_image(self, path, names, imgs):
        raise NotImplementedError
    
    def augmentation(self, imgs):
        raise NotImplementedError

