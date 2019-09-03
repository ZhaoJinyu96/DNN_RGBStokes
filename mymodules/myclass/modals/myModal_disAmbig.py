# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:45:03 2019

@author: 0000145046
"""

# add path
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.joinpath("../../")))

import numpy as np
import cv2

import chainer

from mymodules.myutils.imageutils import MAX_8BIT, MAX_16BIT
import mymodules.myutils.imageutils as iml
from mymodules.myutils.pathutils import extractID


class Dataset_disAmbig(chainer.dataset.DatasetMixin):
    """
    the base class for dataset.
    If you want to iplement a modal, please inherit this class,
    and newly implement the "operation" function.
        
    Arg:
        path_list: 
            list object of file path to train.csv
        bit: 
            the bit number of source image. 8 or 16 is allowed.
        range_max:
            range of data; [0 ~ range_max].
        learn: 
            when learning, True. when testing, False.
        reduce: 
            if you want to use all data, input 1. 
            you want to use only half of your data, input 2.
            (int:x > 0) can be used.
    """
    def __init__(self, path_list, bit, range_max, usenormal, learn, reduce):
        
        self.in_max    = None
        self.range_max = range_max
        self.learn     = learn
        self.pairs     = None
        self.usenormal = usenormal # use_inference or use_gt
        
        if bit == 8:
            self.in_max = MAX_8BIT
        elif bit == 16:
            self.in_max = MAX_16BIT
        else:
            raise ValueError("bit:{} is not allowed to be input.".format(bit))
            
        if learn: # training
            self.pairs = self.learn_pairs(path_list, reduce)
            
        else: # test
            pairs = np.empty((0, 2), dtype=str)
            for path in path_list:

                csv = np.loadtxt(
                    path.joinpath("path.csv"), delimiter=",", dtype=str)
                
                if len(csv.shape) == 1:
                    csv = np.array([csv])
                
                if self.usenormal == False: # no GT
                    for imgnum in range(csv.shape[0]):
                        gtamb_path = str(path.joinpath("gt", csv[imgnum, 3]))
                        s0_path = str(path.joinpath("train", csv[imgnum, 0]))
                        
                        pairs = np.append(
                                pairs,
                                np.array([[gtamb_path, s0_path]]),
                                axis=0)
                        
                else:
                    for imgnum in range(csv.shape[0]):
                        usenormal = pathlib.Path(self.usenormal)
                        imageID = extractID(csv[imgnum, 0])
                        gtamb_path = str(
                                usenormal.joinpath(path.name, "normal", "{}.png".format(imageID))
                                )
                        s0_path = str(path.joinpath("train", csv[imgnum, 0]))
                        
                        pairs = np.append(
                                pairs,
                                np.array([[gtamb_path, s0_path]]),
                                axis=0)
                        
                    
            self.pairs = pairs
            
    def __len__(self):
        return self.pairs.shape[0]
    
    
    def get_example(self, i):
        filename = self.pairs[i, :]
        img_pair = self.get_image(filename)
        
        return img_pair

            
    def get_image(self, filename):
        
        gt_amb = cv2.imread(filename[0], -1).astype(np.float32)
        s0     = self.averageRGB(self.my_imread(filename[1]))
        
        if self.usenormal == False:
            # make ambient normal from gt
            gt_amb = iml.makeAmbNormal(gt_amb[:,:,::-1])
            gt_amb = gt_amb.transpose(2, 0, 1)
            # range
            gt_amb = gt_amb * self.range_max / self.in_max
        else:
            gt_amb = gt_amb[:,:,::-1].transpose(2, 0, 1)
            gt_amb = gt_amb * self.range_max / MAX_8BIT
        
        s0 = s0 * self.range_max / self.in_max
        
        stokes_array = np.array(
                [gt_amb[0,:,:], gt_amb[1,:,:], gt_amb[2,:,:], s0])
        
        if self.learn:
            gt_array   = self.load_gt(filename)
            mask_array = self.load_mask(filename[2])
            
            return stokes_array, gt_array, mask_array
        else:
            return stokes_array
        
    
    def learn_pairs(self, path_list, reduce):
        pairs = np.empty((0, 4), dtype=str)
        
        for csvnum in range(len(path_list)):
            
            path     = path_list[csvnum]
            csv      = np.loadtxt(path.joinpath("path.csv"),
                                      delimiter=",", dtype=str)
            
            for imgnum in range(csv.shape[0]):
                if imgnum % reduce ==0:
                    gtamb_path = str(path.joinpath("gt", csv[imgnum, 3]))
                    s0_path = str(path.joinpath("train", csv[imgnum, 0]))
                    
                    mask_path  = str(path.joinpath("mask" , csv[imgnum, 3]))
                    
                    gt_path = str(path.joinpath("gt", csv[imgnum, 3]))
                    
                    pairs = np.append(
                            pairs,
                            np.array([[gtamb_path, s0_path, mask_path, gt_path]]),
                            axis=0)
    
        return pairs
    
    
    def load_gt(self, filename):
        gt = self.my_imread(filename[3])
        gt = gt * self.range_max / self.in_max
        
        return np.array([gt[2,:,:], gt[1,:,:], gt[0,:,:]])
    
    def load_mask(self, maskname):
        mask = self.my_imread(maskname)
        mask[mask>0] = 1
        
        return np.array([mask[2,:,:], mask[1,:,:], mask[0,:,:]])
    
    
    def my_imread(self, filename):
        return cv2.imread(filename, -1).astype(np.float32).transpose(2, 0, 1)

    def averageRGB(self, img):
        if img.shape[0] != 3:
            raise ValueError("Input img has {}ch".format(img.shape[0]))
        else:
            return (img[0,:,:]+img[1,:,:]+img[2,:,:]) / 3.
