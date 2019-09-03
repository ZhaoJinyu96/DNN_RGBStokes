# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:02:11 2018

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
from mymodules.myutils.polarutils import normalize_s0s1s2


class Dataset_base(chainer.dataset.DatasetMixin):
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
    def __init__(self, path_list, bit, range_max, learn, reduce):
        
        self.in_max    = None
        self.range_max = range_max
        self.learn     = learn
        self.pairs     = None
        

        if bit == 8:
            self.in_max = MAX_8BIT
        elif bit == 16:
            self.in_max = MAX_16BIT
        else:
            raise ValueError("bit:{} is not allowed to be input.".format(bit))
        
        
        if learn: # training
            self.pairs = self.learn_pairs(path_list, reduce)
        
        else: # test
            pairs = np.empty((0, 3), dtype=str)
            for path in path_list:
                csv = np.loadtxt(path.joinpath("path.csv"), delimiter=",", dtype=str)
                
                if len(csv.shape) == 1:
                    csv = np.array([csv])
                    
                for imgnum in range(csv.shape[0]):
                    s0_path = str(path.joinpath("train", csv[imgnum, 0]))
                    s1_path = str(path.joinpath("train", csv[imgnum, 1]))
                    s2_path = str(path.joinpath("train", csv[imgnum, 2]))
                    
                    pairs = np.append(pairs, 
                                      np.array([[s0_path, s1_path, s2_path]]), 
                                      axis=0)
            self.pairs = pairs
    
    
    def __len__(self):
        return self.pairs.shape[0]
    
    
    def get_example(self, i):
        filename = self.pairs[i, :]
        img_pair = self.get_image(filename)
        
        return img_pair 
    
    
    def get_image(self, filename):

        s0 = self.my_imread(filename[0])
        s1 = self.my_imread(filename[1])
        s2 = self.my_imread(filename[2])
        
        s0, s1, s2 = normalize_s0s1s2(s0, s1, s2, self.in_max)
        # Operation
        stokes_array = self.operation(s0, s1, s2)
        
        if self.learn:
            gt_array   = self.load_gt(filename)
            mask_array = self.load_mask(filename[3])
            
            return stokes_array, gt_array, mask_array
        else:
            return stokes_array
        
        
    def operation(self, s0, s1, s2):
        raise NotImplementedError()
    
    def learn_pairs(self, path_list, reduce):
        raise NotImplementedError()
    
    def load_gt(self, filename):
        raise NotImplementedError()
    
    def load_mask(self, maskname):
        raise NotImplementedError()


    def my_imread(self, filename):
        return cv2.imread(filename, -1).astype(np.float32).transpose(2, 0, 1)

    def averageRGB(self, img):
        if img.shape[0] != 3:
            raise ValueError("Input img has {}ch".format(img.shape[0]))
        else:
            return (img[0,:,:]+img[1,:,:]+img[2,:,:]) / 3.
    
    def convertRange_s0DopPhase(self, s0, dop, phase):
        """
        convert range of s0 dop phase.
        In:
            s0 0~1
            dop 0~1
            phase 0~180
        Out:
            s0, dop, phase 0~self.max
        """
        s0 = s0*self.range_max
        dop = dop*self.range_max
        phase = phase*self.range_max /180.
        
        return s0, dop, phase

