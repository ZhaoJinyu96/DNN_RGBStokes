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

from shutil import copy
#import shutil

class Dataset_illumEst_base(myDatasetBase.Dataset_base):
        
    def learn_pairs(self, path_list, reduce):

        pairs = np.empty((0, 5), dtype=str)
        
        for csvnum in range(len(path_list)-1):

            if csvnum % 2 == 1:
                continue
            

            path = path_list[csvnum]
            path_spe = ptl.trainPath2spePath(path)
            csv      = np.loadtxt(path.joinpath("path.csv"),
                                      delimiter=",", dtype=str)

            path_sphere = path_list[csvnum + 1]
            path_sphere_spe = ptl.trainPath2spePath(path_sphere)
            #csv_sphere      = np.loadtxt(path_sphere.joinpath("path.csv"), delimiter=",", dtype=str)
            #shutil.copytree(str(path_sphere_spe.joinpath("train")), str(path_spe.joinpath("gt")))
            
            for imgnum in range(csv.shape[0]):
                if imgnum % reduce ==0:
                    s0_path   = str(path_spe.joinpath("train", csv[imgnum, 0]))
                    s1_path   = str(path_spe.joinpath("train", csv[imgnum, 1]))
                    s2_path   = str(path_spe.joinpath("train", csv[imgnum, 2]))

                    mask_path = str(path.joinpath("mask" , csv[imgnum, 3]))

                    #copy(str(path_sphere_spe.joinpath("train", csv[imgnum, 0])), str(path_spe.joinpath("gt")))
                    #gt_path   = str(path.joinpath("gt" , csv[imgnum, 3]))
                    gt_path = str(path_sphere_spe.joinpath("train", csv[imgnum, 0]))
                    
                    pairs = np.append(pairs, 
                                      np.array([[s0_path, s1_path, s2_path, mask_path, gt_path]]), 
                                      axis=0)
                    
        return pairs
    
    def load_gt(self, filename):
        gt = self.my_imread(filename[4])
        gt = gt * self.range_max / self.in_max
        
        return np.array([gt[2,:,:], gt[1,:,:], gt[0,:,:]])
        
    
    def load_mask(self, maskname):
        mask = self.my_imread(maskname)
        mask[mask>0] = 1       
        return np.array([mask[2,:,:], mask[1,:,:], mask[0,:,:]])

        

class Dataset_s0s1s2(myOperationBase.Operation_s0s1s2,
                     Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_s0s1s2_gray(myOperationBase.Operation_s0s1s2_gray,
                          Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_onlys0(myOperationBase.Operation_onlys0,
                     Dataset_illumEst_base):

    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_onlys0_gray(myOperationBase.Operation_onlys0_gray,
                          Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_onlys0_gray3ch(myOperationBase.Operation_onlys0_gray3ch,
                             Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)

  
class Dataset_s0DopPhase(myOperationBase.Operation_s0DopPhase,
                         Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_s0DopPhase_gray(myOperationBase.Operation_s0DopPhase_gray,
                              Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)


class Dataset_four_polar(myOperationBase.Operation_four_polar,
                         Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)      


class Dataset_four_polar_gray(myOperationBase.Operation_four_polar_gray,
                              Dataset_illumEst_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
