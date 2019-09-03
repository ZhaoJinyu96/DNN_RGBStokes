# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:38:09 2019

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


class DataAugmentation_difSpe_Base(dataAugmentationBaseClass.DataAugmentationBaseClass):
    
    def __init__(self, path_list, bit, operationName):
        super().__init__(path_list, bit, operationName)
        
    
    def make_savedir(self, path):
        # "./hoge/hoge" -> "./hoge/hoge_noise"
        basePath = pathlib.Path(str(path) + "_" + self.operationName)
        
        basePath.joinpath("gt/").mkdir(parents=True)
        basePath.joinpath("gt_difspe/").mkdir(parents=True)
        basePath.joinpath("train/").mkdir(parents=True)
        
        # "./hoge/hoge_noise/gt_difspe/hoge_noise_spe"
        path_spe = basePath.joinpath("gt_difspe/", basePath.name+"_spe")
        path_dif = basePath.joinpath("gt_difspe/", basePath.name+"_dif")
        
        path_spe.joinpath("train/").mkdir(parents=True)
        path_dif.joinpath("train/").mkdir(parents=True)

    
    
    def load_s0s1s2(self, path, names):
        s0_name = path.joinpath("train/",names[0])
        s1_name = path.joinpath("train/",names[1])
        s2_name = path.joinpath("train/",names[2])
        
        s0_img = cv2.imread(str(s0_name), -1).astype(np.float32)
        s1_img = cv2.imread(str(s1_name), -1).astype(np.float32)
        s2_img = cv2.imread(str(s2_name), -1).astype(np.float32)
        
        s0_img, s1_img, s2_img = normalize_s0s1s2(s0_img, s1_img, s2_img, self.max)
        return s0_img, s1_img, s2_img
    
        
    def load_image(self, path, names):
        s0_img, s1_img, s2_img = self.load_s0s1s2(path, names) # NOTICE, they are normalized!!
        
        gt_name = path.joinpath("gt/",names[3])
        gt_img  = cv2.imread(str(gt_name), -1).astype(np.float32)
        
        path_spe = path.joinpath("gt_difspe/", path.name+"_spe")
        path_dif = path.joinpath("gt_difspe/", path.name+"_dif")
        
        s0spe_img, s1spe_img, s2spe_img = self.load_s0s1s2(path_spe, names)
        s0dif_img, s1dif_img, s2dif_img = self.load_s0s1s2(path_dif, names)
        
        return (s0_img,    s1_img,    s2_img,
                gt_img,
                s0spe_img, s1spe_img, s2spe_img,
                s0dif_img, s1dif_img, s2dif_img)

    
    def save_image(self, savepath, names, imgs):
        s0_img, s1_img, s2_img, gt_img, s0spe_img, s1spe_img, s2spe_img, s0dif_img, s1dif_img, s2dif_img = imgs
        
        cv2.imwrite(
                str(savepath.joinpath("train/",names[0])),
                s0_img.astype(self.bit))
        cv2.imwrite(
                str(savepath.joinpath("train/",names[1])),
                s1_img.astype(self.bit))
        cv2.imwrite(
                str(savepath.joinpath("train/",names[2])),
                s2_img.astype(self.bit))
        cv2.imwrite(
                str(savepath.joinpath("gt/",names[3])),
                gt_img.astype(self.bit))
        
        path_spe = savepath.joinpath("gt_difspe/", savepath.name+"_spe")
        path_dif = savepath.joinpath("gt_difspe/", savepath.name+"_dif")
        
        cv2.imwrite(
                str(path_spe.joinpath("train/",names[0])),
                s0spe_img.astype(self.bit))
        cv2.imwrite(
                str(path_spe.joinpath("train/",names[1])),
                s1spe_img.astype(self.bit))
        cv2.imwrite(
                str(path_spe.joinpath("train/" + names[2])),
                s2spe_img.astype(self.bit))
        cv2.imwrite(
                str(path_dif.joinpath("train/",names[0])),
                s0dif_img.astype(self.bit))
        cv2.imwrite(
                str(path_dif.joinpath("train/",names[1])),
                s1dif_img.astype(self.bit))
        cv2.imwrite(
                str(path_dif.joinpath("train/",names[2])),
                s2dif_img.astype(self.bit))
        

        
# augmentations
class FlipImage_Horizontal(DataAugmentation_difSpe_Base):
    def __init__(self, path_list, bit, operationName):
        super().__init__(path_list, bit, operationName)
    
    def augmentation(self, imgs):
        s0, s1, s2, gt, s0spe, s1spe, s2spe, s0dif, s1dif, s2dif = imgs
        
        # flip coordinate of gt image
        gt[:,:,2] = gt[:,:,2] * (-1) + self.max
        gt[:,:,2][gt[:,:,0]==0] = 0 # z=0 means background area
        
        # flip images
        s0    = s0[:,::-1,:]
        s1    = s1[:,::-1,:]
        s2    = s2[:,::-1,:] * -1
        s0, s1, s2 = de_normalize_s0s1s2(s0, s1, s2, self.max)
        
        gt = gt[:,::-1,:]
        
        s0spe = s0spe[:,::-1,:]
        s1spe = s1spe[:,::-1,:]
        s2spe = s2spe[:,::-1,:] * -1
        s0spe, s1spe, s2spe = de_normalize_s0s1s2(s0spe, s1spe, s2spe, self.max)
        
        s0dif = s0dif[:,::-1,:]
        s1dif = s1dif[:,::-1,:]
        s2dif = s2dif[:,::-1,:] * -1
        s0dif, s1dif, s2dif = de_normalize_s0s1s2(s0dif, s1dif, s2dif, self.max)
        
        return (s0, s1, s2,
                gt,
                s0spe, s1spe, s2spe,
                s0dif, s1dif, s2dif)
        

class Blur(DataAugmentation_difSpe_Base):
    def __init__(self, path_list, bit, operationName, square):
        super().__init__(path_list, bit, operationName)
        self.square = square
        
    def augmentation(self, imgs):
        s0, s1, s2, gt, s0spe, s1spe, s2spe, s0dif, s1dif, s2dif = imgs        
        s0 = np.clip(cv2.blur(s0, self.square), 0, 1)
        s1 = np.clip(cv2.blur(s1, self.square), -1, 1)
        s2 = np.clip(cv2.blur(s2, self.square), -1, 1)
        
        s0, s1, s2 = de_normalize_s0s1s2(s0, s1, s2, self.max)
        s0spe, s1spe, s2spe = de_normalize_s0s1s2(s0spe, s1spe, s2spe, self.max)
        s0dif, s1dif, s2dif = de_normalize_s0s1s2(s0dif, s1dif, s2dif, self.max)
        
        return (s0, s1, s2,
                gt,
                s0spe, s1spe, s2spe,
                s0dif, s1dif, s2dif)
        

class Noise(DataAugmentation_difSpe_Base):
    def __init__(self, path_list, bit, operationName, sigma):
        super().__init__(path_list, bit, operationName)
        self.sigma = sigma
        
    def augmentation(self, imgs):
        s0, s1, s2, gt, s0spe, s1spe, s2spe, s0dif, s1dif, s2dif = imgs        
        
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
        s0spe, s1spe, s2spe = de_normalize_s0s1s2(s0spe, s1spe, s2spe, self.max)
        s0dif, s1dif, s2dif = de_normalize_s0s1s2(s0dif, s1dif, s2dif, self.max)
        
        return (s0, s1, s2,
                gt,
                s0spe, s1spe, s2spe,
                s0dif, s1dif, s2dif)
        

class Intensity(DataAugmentation_difSpe_Base):
    def __init__(self, path_list, bit, operationName):
        super().__init__(path_list, bit, operationName)
        
    def augmentation(self, imgs):
        s0, s1, s2, gt, s0spe, s1spe, s2spe, s0dif, s1dif, s2dif = imgs
        
        """mn = np.mean(s0) * self.max
        newmn = np.random.normal(mn, mn/4)
        gain = newmn / mn
        if gain < 0:
            raise ValueError("gain should be larger than 0!")"""
        
        # random number [a, b)
        a, b = 0.5, 1.5
        gain = (b - a) * np.random.rand() + a
        
        # reconstruct cosine curve
        i_0, i_45, i_90, i_135 = calc_fourPolar_from_stokes(s0, s1, s2)
        i_0spe, i_45spe, i_90spe, i_135spe = calc_fourPolar_from_stokes(
                s0spe, s1spe, s2spe)
        i_0dif, i_45dif, i_90dif, i_135dif = calc_fourPolar_from_stokes(
                s0dif, s1dif, s2dif)
        
        # multiply gain
        i_0 *= gain
        i_45 *= gain
        i_90 *= gain
        i_135 *= gain
        
        i_0spe *= gain
        i_45spe *= gain
        i_90spe *= gain
        i_135spe *= gain
        
        i_0dif *= gain
        i_45dif *= gain
        i_90dif *= gain
        i_135dif *= gain
        
        # return back to s0s1s2
        s0 = i_0 + i_90
        s1 = i_0 - i_90
        s2 = i_45 - i_135
        
        s0spe = i_0spe + i_90spe
        s1spe = i_0spe - i_90spe
        s2spe = i_45spe - i_135spe
        
        s0dif = i_0dif + i_90dif
        s1dif = i_0dif - i_90dif
        s2dif = i_45dif - i_135dif
        
        # clipping
        s0 = np.clip(s0, 0, 1)
        s1 = np.clip(s1, -1, 1)
        s2 = np.clip(s2, -1, 1)
        
        s0spe = np.clip(s0spe, 0, 1)
        s1spe = np.clip(s1spe, -1, 1)
        s2spe = np.clip(s2spe, -1, 1)
        
        s0dif = np.clip(s0dif, 0, 1)
        s1dif = np.clip(s1dif, -1, 1)
        s2dif = np.clip(s2dif, -1, 1)
        
        # denormalize
        s0, s1, s2 = de_normalize_s0s1s2(s0, s1, s2, self.max)
        s0spe, s1spe, s2spe = de_normalize_s0s1s2(s0spe, s1spe, s2spe, self.max)
        s0dif, s1dif, s2dif = de_normalize_s0s1s2(s0dif, s1dif, s2dif, self.max)
        
        return (s0, s1, s2,
                gt,
                s0spe, s1spe, s2spe,
                s0dif, s1dif, s2dif)
        

