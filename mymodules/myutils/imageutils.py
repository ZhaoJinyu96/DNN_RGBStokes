# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:20:28 2019

@author: 0000145046
"""

import numpy as np
import cv2


MAX_8BIT  = 255.
MAX_16BIT = 65535.


def assign_max_8or16bit(img):
    """
    return MAX_8BIT or MAX_16BIT depending on 
    whether input image is 8bit or 16bit.
    
    In:
        img: numpy.ndarray. should be 0~255 or 0~65535.
    Out:
        MAX_8BIT, if 1 < max(input) <= 255.
        MAX_16BIT, if 255 < max(input) <= 65535.
        
        This function will raise Value Error,
        if max(input) < 1 or 65535 < max(input).
    """
    maxval = np.max(img)
    
    if MAX_8BIT < maxval <= MAX_16BIT:
        return MAX_16BIT
    elif 1 < maxval <= MAX_8BIT:
        return MAX_8BIT
    else:
        raise ValueError("max(input) is wrong.".format(maxval))
    

def normalize(N):
    # N (x, y, 3)
    N_ = np.copy(N)
    
    mid = assign_max_8or16bit(N) / 2
        
    N_ -= mid
    
    N_norm = np.sqrt(np.sum(N_*N_, axis=2))
    invalid = (N_norm == 0)
    N_norm[invalid] += 0.0001
    
    N_[:,:,0] /= N_norm
    N_[:,:,1] /= N_norm
    N_[:,:,2] /= N_norm
    
    N_[N_norm==0] = 0
    
    return N_


def calcAngError(gt, recon):
    invalid = (np.sum(gt*gt, axis=2) == 0)
    
    gt_dot_recon = np.sum(normalize(gt)*normalize(recon), axis=2)
    gt_dot_recon = np.clip(gt_dot_recon, -1, 1)
    
    ang_deg = np.rad2deg(np.arccos(gt_dot_recon))
    ang_deg[invalid] = 0
    
    return ang_deg


def shrinkeMask_byEdge(mask):
    
    maskcpy = np.copy(mask)
    
    maskcpy[0,:]  = 0
    maskcpy[-1,:] = 0
    maskcpy[:,0]  = 0
    maskcpy[:,-1] = 0
    
    h ,w = mask.shape[0], mask.shape[1]
    for i in range(1, h-1):
        for j in range(1, w-1):
            var = mask[(i-1,i,i+1), j]
            hor = mask[i, (j-1,j,j+1)]
            
            if np.sum(var)==2 and var[1]==1: # edge
                maskcpy[i, j] = 0
            elif np.sum(hor)==2 and hor[1]==1:
                maskcpy[i, j] = 0
            elif np.sum(var)==1 and var[1]==1:
                maskcpy[i, j] = 0
            elif np.sum(hor)==1 and hor[1]==1:
                maskcpy[i, j] = 0
                
    return maskcpy


def extractObjArea_edgeRemoved(err, mask, shrinkNum=1):
    maxval = assign_max_8or16bit(mask)
    
    mask = (mask[:,:,1] / maxval)
    
    for cnt in range(shrinkNum):
        mask = shrinkeMask_byEdge(mask)
    
    err = err[mask.astype(bool)]
    
    return err
    

def extractObjArea(err, mask):
    maxval = assign_max_8or16bit(mask)
    
    mask = (mask[:,:,1] / maxval)
    err = err[mask.astype(bool)]
    
    return err


def extractGroundArea(err, mask):
    maxval = assign_max_8or16bit(mask)
    
    mask = (mask[:,:,0] / maxval)
    err = err[mask.astype(bool)]
    
    return err


def extractMaskArea(err, mask, maskFlag):
    
    if maskFlag == "all":
        return err
    
    elif maskFlag == "object":
        return extractObjArea(err, mask)
        #return extractObjArea_edgeRemoved(err, mask, 3)
    
    elif maskFlag == "ground":
        return extractGroundArea(err, mask)


def makeAmbNormal(N):
    """
    convert GT normal map into 180 ambiguity normal map.
    
    Args:
        N: normal map, RGB correspond to xyz.
    return:
        ambN: normal map which has 180deg ambiguity.
              RGB correspond to xyz.
    """
    maxnum = assign_max_8or16bit(N)
    
    N = (N / maxnum) * 2 - 1 # (-1, 1)
    
    x, y, z = N[:,:,0], N[:,:,1], N[:,:,2]
    
    theta = np.rad2deg(np.arccos(z))
    phi   = np.sign(y) * np.rad2deg(np.arccos(x / np.sqrt(x*x+y*y)))
    phi[phi < 0] = phi[phi < 0] + 180     # Degree [0, 360)
    
    amb_x = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    amb_y = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    
    ambN = cv2.merge((amb_x, amb_y, z))
    
    ambN = (ambN + 1) / 2. * maxnum
    
    return ambN


def calcZenithDegMap(normal_map):
    """
    calculate zenith map from normal map.
    
    Args:
        normal_map:
            numpy.ndarray (w, h, (x, y, z))
    Out:
        zenith map. numpy.ndarray (w, h).
        written in degree, not rad.
    """
    normal_map = normalize(normal_map)
    theta = np.arccos(normal_map[:,:,2])
    
    return np.rad2deg(theta)
    

def normalErr2depthErr(zenith_deg_gt, ang_deg_err, pixN=500e4/2000, Z=3000, fovdeg=50):
    """
    convert normal error to depth error.
    
    Args:
        zenith_deg_gt:
            numpy.ndarray of (h, w).
            zenith angle of gt normal map.
        ang_deg_err:
            numpy.ndarray of (h, w).
            Error map of furface normal.
            Should be written in degree, not rad.
        pixN:
            The number of pixels about horizontal line.
            Default value is 500e4/2000.
        Z:
            Distance between a camera and objects.
            Default value is 3000[m].
        fovdeg:
            Field of horizontal view.
            Should be written in degree, not rad.
            Default value is 50.
    Out:
        numpy.ndarray of (h, w).
        Error map of depth.
    """
    K = Z * np.tan(np.deg2rad(fovdeg)/2.) / (pixN/2.) 
    tan_gt = np.tan(np.deg2rad(zenith_deg_gt))
    tan_err = np.tan(np.deg2rad(ang_deg_err))
    
    eps = 1e-06
    depth_err1 = K * (1 + tan_gt*tan_gt) * tan_err / (1 - tan_gt*tan_err + eps)
    depth_err1 = np.abs(depth_err1)
    
    depth_err2 = K * (1 + tan_gt*tan_gt) * tan_err / (1 + tan_gt*tan_err + eps)
    depth_err2 = np.abs(depth_err2)
    
    mask = (depth_err2 > depth_err1)
    depth_err1[mask] = depth_err2[mask]
    
    return depth_err1


def calcPixNfromFoVDepthErr(fovdeg, zenith_deg_gt, ang_deg_err, mask, depth_err=2, Z=3000):
    tan_gt = np.tan(np.deg2rad(zenith_deg_gt))
    tan_err = np.tan(np.deg2rad(ang_deg_err))
    
    eps = 1e-06
    M = (1 + tan_gt*tan_gt) * tan_err / (1 - tan_gt*tan_err + eps)
    M = np.abs(M)
    M_ = (1 + tan_gt*tan_gt) * tan_err / (1 + tan_gt*tan_err + eps)
    M_ = np.abs(M_)
    
    mask_comp = (M_ > M)
    M[mask_comp] = M_[mask_comp]
    
    pixN = 2 * np.mean(M[mask]) * Z * np.tan(np.deg2rad(fovdeg/2.)) / depth_err
    
    return pixN
