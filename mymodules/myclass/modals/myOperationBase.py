# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:26:19 2019

@author: 0000145046
"""

# add path
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.joinpath("../../")))

import numpy as np

import mymodules.myclass.modals.myDatasetBase as myDatasetBase
import mymodules.myutils.polarutils as pol



class Operation_s0s1s2(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        s0, s1, s2 = pol.de_normalize_s0s1s2(s0, s1, s2, self.range_max)
        
        stokes_array = np.array([s0[2,:,:], s0[1,:,:], s0[0,:,:],
                                 s1[2,:,:], s1[1,:,:], s1[0,:,:],
                                 s2[2,:,:], s2[1,:,:], s2[0,:,:]])
        
        return stokes_array


class Operation_s0s1s2_gray(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        s0 = self.averageRGB(s0)
        s1 = self.averageRGB(s1)
        s2 = self.averageRGB(s2)
        
        s0, s1, s2 = pol.de_normalize_s0s1s2(s0, s1, s2, self.range_max)
        
        return np.array([s0, s1, s2])


class Operation_onlys0(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        s0 = s0 * self.range_max
        return np.array([s0[2,:,:], s0[1,:,:], s0[0,:,:]])


class Operation_onlys0_gray(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        s0 = self.averageRGB(s0)
        s0 = s0 * self.range_max
        
        return np.array([s0])


class Operation_onlys0_gray3ch(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        s0 = self.averageRGB(s0)
        s0 = s0 * self.range_max
        
        return np.array([s0, s0, s0])
    
  
class Operation_s0DopPhase(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)

    
    def operation(self, s0, s1, s2):
        
        dop   = pol.calc_dop_from_stokes(s0, s1, s2)
        phase = pol.calc_polar_phase_from_stokes(s1, s2)
        
        s0, dop, phase = self.convertRange_s0DopPhase(s0, dop, phase)
        
        stokes_array = np.array([   s0[2,:,:],    s0[1,:,:],    s0[0,:,:],
                                   dop[2,:,:],   dop[1,:,:],   dop[0,:,:],
                                 phase[2,:,:], phase[1,:,:], phase[0,:,:]])
    
        return stokes_array


class Operation_s0DopPhase_gray(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        
        dop   = pol.calc_dop_from_stokes(s0, s1, s2)
        phase = pol.calc_polar_phase_from_stokes(s1, s2)
        
        s0, dop, phase = self.convertRange_s0DopPhase(s0, dop, phase)
        
        s0 = self.averageRGB(s0)
        dop = self.averageRGB(dop)
        phase = self.averageRGB(phase)
        
        return np.array([s0, dop, phase])
  

class Operation_four_polar(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        
        i_0, i_45, i_90, i_135 = pol.calc_fourPolar_from_stokes(s0, s1, s2)
        
        i_0   *= self.range_max
        i_45  *= self.range_max
        i_90  *= self.range_max
        i_135 *= self.range_max
        
        stokes_array = np.array([  i_0[2,:,:],   i_0[1,:,:],   i_0[0,:,:],
                                  i_45[2,:,:],  i_45[1,:,:],  i_45[0,:,:],
                                  i_90[2,:,:],  i_90[1,:,:],  i_90[0,:,:],
                                 i_135[2,:,:], i_135[1,:,:], i_135[0,:,:]])
    
        return stokes_array    


class Operation_four_polar_gray(myDatasetBase.Dataset_base):
    
    def __init__(self, path_list, bit, range_max, learn, reduce):
        super().__init__(path_list, bit, range_max, learn, reduce)
        
    def operation(self, s0, s1, s2):
        
        i_0, i_45, i_90, i_135 = pol.calc_fourPolar_from_stokes(s0, s1, s2)
        
        i_0   = self.averageRGB(i_0) * self.range_max
        i_45  = self.averageRGB(i_45) * self.range_max
        i_90  = self.averageRGB(i_90) * self.range_max
        i_135 = self.averageRGB(i_135) * self.range_max
        
        return np.array([i_0, i_45, i_90, i_135])

