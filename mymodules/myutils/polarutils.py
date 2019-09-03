# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:24:52 2019

@author: 0000145046
"""

# add path
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))

import numpy as np
import cv2

from imageutils import assign_max_8or16bit

    
def normalize_s0s1s2(s0, s1, s2, maxVal):
    """
    s0 is normalized into (0, 1)
    s1 and s2 are normalized into (-1, 1)
    """
    s0 = s0 / maxVal          #(0, 1)
    s1 = (s1 / maxVal)*2 - 1  #(-1, 1)
    s2 = (s2 / maxVal)*2 - 1  #(-1, 1)
    
    return (s0, s1, s2)

def de_normalize_s0s1s2(s0, s1, s2, maxVal):
    """
    Input s0, s1, s2 should be:
        s0; (0, 1)
        s1; (-1, 1)
        s2; (-1, 1)
    """
    s0 = s0 * maxVal
    s1 = (s1 + 1) * maxVal / 2.
    s2 = (s2 + 1) * maxVal / 2.
    
    return (s0, s1, s2)


def normalize_check(img, minVal, maxVal):
    img_min, img_max = np.min(img), np.max(img)
    return minVal <= img_min and img_max <= maxVal


def normalize_check_s0s1s2(s0, s1, s2):
    """
    Check the range of s0, s1 and s2.
    Out:
        True: (0<=s0<=1 && -1<=s1,s2<=1)
        False: not the above case.
    """
    if (normalize_check(s0, 0, 1)) and \
           (normalize_check(s1, -1, 1)) and\
               (normalize_check(s2, -1, 1)):
        return True

    else:
        return False
    
def normalize_check_s1s2(s1, s2):
    if (normalize_check(s1, -1, 1)) and\
           (normalize_check(s2, -1, 1)):
        return True
    else:
        return False


def calc_fourPolar_from_stokes(s0, s1, s2):
    """
    Input s0, s1, s2 should be:
        s0; (0, 1)
        s1; (-1, 1)
        s2; (-1, 1)
    """
    if not normalize_check_s0s1s2(s0, s1, s2):
        raise ValueError("s0, s1, s2 are not normalized.")
        
    i_0 = (s0 + s1) / 2. 
    i_45 = (s0 + s2) / 2.
    i_90 = (s0 - s1) / 2.
    i_135 = (s0 - s2) / 2.
    
    return i_0, i_45, i_90, i_135


def calc_s0s1s2_from_fourPolar(i0, i45, i90, i135):
    """
    Input images should be normalized in (0,1).
    """
    s0 = i0 + i90     # (0, 1)
    s1 = i0 - i90     # (-1, 1)
    s2 = i45 - i135   # (-1, 1)

    return s0, s1, s2


def calc_dop_from_stokes(s0, s1, s2):
    """ 
    calculate dop from s0, s1, s2
    s0, s1 and s2 should be normalized into
    (0,1), (-1,1), (-1,1)
    In:
        s0s1s2
    Out:
        dop, 0~1
    """
    if not normalize_check_s0s1s2(s0, s1, s2):
        raise ValueError("s0, s1, s2 are not normalized.")
    
    invalid = (s0==0)
    s0[invalid] += 0.0001
    
    dop = np.sqrt(s1**2 + s2**2) / s0   #(0, 1)
    dop[invalid] = 0
    dop = np.clip(dop, 0.0, 1.0)
    
    return dop


def calc_polar_phase_from_stokes(s1, s2):
    """ 
    calculate dop from s0, s1, s2
    s0, s1 and s2 should be normalized into
    (0,1), (-1,1), (-1,1)
    
    out:
        phase: phase of polarization, deg, [0, 180)
    """
    if not normalize_check_s1s2(s1, s2):
        raise ValueError("s1, s2 are not normalized.")
        
    invalid = (s1 == 0) & (s2 == 0)
    phase = np.rad2deg(np.arctan2(s2, s1))   # Degree (-180, 180]
    phase[phase < 0] = phase[phase < 0] + 360      # Degree [0, 360)
    phase = phase / 2.                              # Degree [0, 180)
    phase[invalid] = 0
    
    phase = np.clip(phase, 0, 180)
    
    return phase


def calc_diffuse_dop_from_zenith_deg(zenith_deg, ref_id):

    sin_val = np.sin(np.deg2rad(zenith_deg))
    cos_val = np.cos(np.deg2rad(zenith_deg))

    difs_dop = (ref_id - 1/ref_id)**2 * sin_val**2 / \
               (2+2*ref_id**2-(ref_id+1/ref_id)**2 * sin_val**2 + 4 * cos_val * np.sqrt(ref_id**2 - sin_val**2))

    return difs_dop

def calc_diffuse_zenith_from_dop(in_dop, ref_id):
    
    max_dop = calc_diffuse_dop_from_zenith_deg(90, ref_id)
    in_dop[in_dop > max_dop] = max_dop

    val_A = 2 * (1 - in_dop) - (1 + in_dop) * (ref_id**2 + 1 / (ref_id ** 2))
    val_B = 4 * in_dop
    val_C = 1 + ref_id**2
    val_D = 1 - ref_id**2

    sin_val_sqrt_num = -1 * val_B * (val_C * (val_A + val_B) -
                                     np.sqrt(val_C**2 * (val_A + val_B)**2 - val_D**2 * (val_A**2 - val_B**2)))
    sin_val_sqrt_denum = 2 * (val_A**2 - val_B**2)

    sin_val = sin_val_sqrt_num / sin_val_sqrt_denum
    sin_val = np.clip(sin_val, 0, 1)
    
    sin_val_mask = (sin_val == 0)
    sin_val[sin_val_mask] += 0.0001
    
    out_zenith = np.rad2deg(np.arcsin(np.sqrt(sin_val)))
    out_zenith[sin_val_mask] = 0
    out_zenith = np.clip(out_zenith, 0, 90)

    return out_zenith
    
def calc_normal_from_dop_and_phase(im_dop, im_phase):
    
    zenith = calc_diffuse_zenith_from_dop(im_dop, 1.5)

    normal_x = np.cos(np.deg2rad(im_phase)) * np.sin(np.deg2rad(zenith))
    normal_y = np.sin(np.deg2rad(im_phase)) * np.sin(np.deg2rad(zenith))
    normal_z = np.cos(np.deg2rad(zenith))

    out_normal = cv2.merge((normal_z, normal_y, normal_x))
    
    out_normal255 = (out_normal + 1) / 2
    out_normal255 = np.clip(out_normal255, 0, 1)
    out_normal255 = out_normal255 * 255
    out_normal255 = out_normal255.astype(np.uint8)
    
    return out_normal255


def calc_PolarImg_fromS0DopPhase(s0, dop, phase, polar_deg):
    """
    Calculate the polar image of designated polarizer angle.
    
    Args:
        s0       : s0 image. The data type should be numpy.float32.
        dop      : dop image.
        phase    : phase image.
        polar_deg: The angle of polarizer. degree (0~180)
        
    Out:
        Polar image.
    """
    maxim = assign_max_8or16bit(s0)
    
    dop = dop / maxim
    phase_deg = phase * 180 / maxim
    
    return (s0 / 2.) + dop * s0 * np.cos(2 * (np.deg2rad(polar_deg) - np.deg2rad(phase_deg))) / 2.
    
    
    
