# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:01:06 2019

@author: 0000145046
"""

# add path
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.joinpath("../../")))

import cv2
import numpy as np

import mymodules.myclass.augmentations.dataAugmentationBaseClass as dataAugmentationBaseClass

from mymodules.myutils.polarutils import normalize_s0s1s2
from mymodules.myutils.polarutils import de_normalize_s0s1s2
from mymodules.myutils.polarutils import calc_fourPolar_from_stokes
from mymodules.myutils.polarutils import calc_s0s1s2_from_fourPolar


class DataAugmentation_normEst_Base(dataAugmentationBaseClass.DataAugmentationBaseClass):
    
    def __init__(self, path_list, bit, operationName):
        super().__init__(path_list, bit, operationName)
    
    def make_savedir(self, path):
        # "./hoge/hoge" -> "./hoge/hoge_noise"
        basePath = pathlib.Path(str(path) + "_" + self.operationName)
        
        basePath.joinpath("gt/").mkdir(parents=True)
        basePath.joinpath("gt_difspe/").mkdir(parents=True)
        basePath.joinpath("train/").mkdir(parents=True)
    
    def load_image(self, path, names):
        s0_name = path.joinpath("train/",names[0])
        s1_name = path.joinpath("train/",names[1])
        s2_name = path.joinpath("train/",names[2])
        gt_name = path.joinpath("gt/",names[3])

        s0_img = cv2.imread(str(s0_name), -1).astype(np.float32)
        s1_img = cv2.imread(str(s1_name), -1).astype(np.float32)
        s2_img = cv2.imread(str(s2_name), -1).astype(np.float32)
        gt_img = cv2.imread(str(gt_name), -1).astype(np.float32)
        
        s0_img, s1_img, s2_img = normalize_s0s1s2(s0_img, s1_img, s2_img, self.max)
        return (s0_img, s1_img, s2_img, gt_img)

    
    def save_image(self, savepath, names, imgs):
        s0_img, s1_img, s2_img, gt_img = imgs
        
        cv2.imwrite(str(savepath.joinpath("train",names[0])), s0_img.astype(self.bit))
        cv2.imwrite(str(savepath.joinpath("train",names[1])), s1_img.astype(self.bit))
        cv2.imwrite(str(savepath.joinpath("train",names[2])), s2_img.astype(self.bit))
        cv2.imwrite(str(savepath.joinpath("gt",names[3])), gt_img.astype(self.bit))



# augmentations
class FlipImage_Horizontal(DataAugmentation_normEst_Base):
    def __init__(self, path_list, bit, operationName):
        super().__init__(path_list, bit, operationName)
            
    def augmentation(self, imgs):
        s0_img, s1_img, s2_img, gt_img = imgs
        
        # flip coordinate of gt image
        gt_img[:,:,2] = gt_img[:,:,2] * (-1) + self.max
        gt_img[:,:,2][gt_img[:,:,0]==0] = 0 # z=0 means background area
        
        # flip images
        s0_img = s0_img[:,::-1,:]
        s1_img = s1_img[:,::-1,:]
        s2_img = s2_img[:,::-1,:] * -1
        gt_img = gt_img[:,::-1,:]
        
        s0_img, s1_img, s2_img = de_normalize_s0s1s2(s0_img, s1_img, s2_img, self.max)
        return (s0_img, s1_img, s2_img, gt_img)
    

class Blur(DataAugmentation_normEst_Base):
    def __init__(self, path_list, bit, operationName, square):
        super().__init__(path_list, bit, operationName)
        self.square = square
        
    def augmentation(self, imgs):
        s0, s1, s2, gt = imgs
        
        s0 = np.clip(cv2.blur(s0, self.square), 0, 1)
        s1 = np.clip(cv2.blur(s1, self.square), -1, 1)
        s2 = np.clip(cv2.blur(s2, self.square), -1, 1)
        
        s0, s1, s2 = de_normalize_s0s1s2(s0, s1, s2, self.max)       
        return (s0, s1, s2, gt)
    

class Noise(DataAugmentation_normEst_Base):
    def __init__(self, path_list, bit, operationName, sigma):
        super().__init__(path_list, bit, operationName)
        self.sigma = sigma
        
    def augmentation(self, imgs):
        s0, s1, s2, gt = imgs
        
        # reconstruct cosine curve
        i_0, i_45, i_90, i_135 = calc_fourPolar_from_stokes(s0, s1, s2)
        i_list = np.array([i_0, i_45, i_90, i_135])
        
        # add gaussian noise
        row, col, ch = s0.shape
        for inum in range(i_list.shape[0]):
            gauss = np.random.normal(0, self.sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            i_list[inum] += gauss
            
        # calc new stokes images
        i_0 = i_list[0]
        i_45 = i_list[1]
        i_90 = i_list[2]
        i_135 = i_list[3]
        
        s0, s1, s2 = calc_s0s1s2_from_fourPolar(i_0, i_45, i_90, i_135)
        s0 = np.clip(s0, 0, 1)
        s1 = np.clip(s1, -1, 1)
        s2 = np.clip(s2, -1, 1)
        
        s0, s1, s2 = de_normalize_s0s1s2(s0, s1, s2, self.max)
        return (s0, s1, s2, gt)
    
 