# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:54:25 2018

@author: 0000145046
"""
# add path
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.joinpath("../../")))

from joblib import Parallel, delayed
import numpy as np

import mymodules.myclass.modals.myDatasetBase as myDatasetBase
import mymodules.myclass.modals.myOperationBase as myOperationBase

import mymodules.myutils.polarutils as pol
import mymodules.myutils.pathutils as ptl


def learn_pairs_sub(path, reduce):
    pairs = np.empty((0, 10), dtype=str)
    
    path_spe = ptl.trainPath2spePath(path)
    path_dif = ptl.trainPath2difPath(path)
    
    csv      = np.loadtxt(path.joinpath("path.csv"),
                              delimiter=",", dtype=str)
    
    for imgnum in range(csv.shape[0]):
        if imgnum % reduce ==0:
            s0_path    = str(path.joinpath("train", csv[imgnum, 0]))
            s1_path    = str(path.joinpath("train", csv[imgnum, 1]))
            s2_path    = str(path.joinpath("train", csv[imgnum, 2]))
            
            mask_path  = str(path.joinpath("mask" , csv[imgnum, 3]))
            
            s0_pathspe = str(path_spe.joinpath("train", csv[imgnum, 0]))
            s1_pathspe = str(path_spe.joinpath("train", csv[imgnum, 1]))
            s2_pathspe = str(path_spe.joinpath("train", csv[imgnum, 2]))
            
            s0_pathdif = str(path_dif.joinpath("train", csv[imgnum, 0]))
            s1_pathdif = str(path_dif.joinpath("train", csv[imgnum, 1]))
            s2_pathdif = str(path_dif.joinpath("train", csv[imgnum, 2]))
            
            
            pairs = np.append(pairs, 
                              np.array([[s0_path   , s1_path   , s2_path,
                                         mask_path,
                                         s0_pathspe, s1_pathspe, s2_pathspe, 
                                         s0_pathdif, s1_pathdif, s2_pathdif]]), 
                              axis=0)
            
    return pairs 

class Dataset_speDif_base(myDatasetBase.Dataset_base):
    
    def learn_pairs(self, path_list, reduce):
        sub_pairs = Parallel(n_jobs=-1)(
                [delayed(learn_pairs_sub)(path, reduce) for path in path_list]
                )
        
        pairs = np.empty((0, 10), dtype=str)
        for sub_pair in sub_pairs:
            pairs = np.append(pairs, sub_pair, axis=0)
            
        return pairs
    
    
    def load_gt(self, filename):
        s0_spe = self.averageRGB(self.my_imread(filename[4]))
        s1_spe = self.averageRGB(self.my_imread(filename[5]))
        s2_spe = self.averageRGB(self.my_imread(filename[6]))
    
        s0_dif = self.averageRGB(self.my_imread(filename[7]))
        s1_dif = self.averageRGB(self.my_imread(filename[8]))
        s2_dif = self.averageRGB(self.my_imread(filename[9]))
        
        s0_spe, s1_spe, s2_spe = pol.normalize_s0s1s2(s0_spe, s1_spe, s2_spe, self.in_max)
        s0_dif, s1_dif, s2_dif = pol.normalize_s0s1s2(s0_dif, s1_dif, s2_dif, self.in_max)
    
        dop_spe   = pol.calc_dop_from_stokes(s0_spe, s1_spe, s2_spe)
        phase_spe = pol.calc_polar_phase_from_stokes(s1_spe, s2_spe)
        
        dop_dif   = pol.calc_dop_from_stokes(s0_dif, s1_dif, s2_dif)
        phase_dif = pol.calc_polar_phase_from_stokes(s1_dif, s2_dif)
        
        s0_spe, dop_spe, phase_spe = self.convertRange_s0DopPhase(s0_spe, dop_spe, phase_spe)
        s0_dif, dop_dif, phase_dif = self.convertRange_s0DopPhase(s0_dif, dop_dif, phase_dif)
        
        return np.array([s0_spe, dop_spe, phase_spe,
                         s0_dif, dop_dif, phase_dif])
        
    
    def load_mask(self, maskname):
        mask = self.my_imread(maskname)
        mask[mask>0] = 1
    
        return np.array([mask[2,:,:], mask[1,:,:], mask[0,:,:],
                         mask[2,:,:], mask[1,:,:], mask[0,:,:]])
            


class Dataset_s0s1s2(myOperationBase.Operation_s0s1s2,
                     Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_s0s1s2_gray(myOperationBase.Operation_s0s1s2_gray,
                          Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_onlys0(myOperationBase.Operation_onlys0,
                     Dataset_speDif_base):

    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_onlys0_gray(myOperationBase.Operation_onlys0_gray,
                          Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_onlys0_gray3ch(myOperationBase.Operation_onlys0_gray3ch,
                             Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)

  
class Dataset_s0DopPhase(myOperationBase.Operation_s0DopPhase,
                         Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_s0DopPhase_gray(myOperationBase.Operation_s0DopPhase_gray,
                              Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)



class Dataset_four_polar(myOperationBase.Operation_four_polar,
                         Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)      


class Dataset_four_polar_gray(myOperationBase.Operation_four_polar_gray,
                              Dataset_speDif_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
