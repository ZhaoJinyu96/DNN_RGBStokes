# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:51:16 2019

@author: 0000145046
"""

# temporaly add this directory to pythonpath
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))


def lossFunselect(lossname):
    import myLossfun
    
    if lossname == "masked_mean_squared_error":
        return myLossfun.masked_mean_squared_error
    elif lossname == "masked_mean_squared_error_difspeWeight":
        return myLossfun.masked_mean_squared_error_difspeWeight
    elif lossname == "masked_MSE_gradient_error":
        return myLossfun.masked_MSE_gradient_error
    elif lossname == "masked_dot_product_error":
        return myLossfun.masked_mean_squared_error
    else:
        return None
    

def networkSelect(networkname, task):
    import networks
    
    if task == "normal_estimation":
        out_size = 3
    elif task == "dif_spe_separation":
        out_size = 6
    elif task == "normal_disAmbiguate":
        out_size = 3
    elif task == "illumination_estimation":
        out_size = 3
        
    
    if networkname == "ResNet18":
        return networks.ResNet18(out_size)
    elif networkname == "ResNet34":
        return networks.ResNet34(out_size)
    elif networkname  == "ResNet34_UpProj":
        return networks.ResNet34_UpProj(out_size)
    elif networkname == "PBR_CVPR2017":
        return networks.PBR_CVPR2017(out_size)
    elif networkname == "PBR_CVPR2017_mod":
        return networks.PBR_CVPR2017_mod(out_size)
    elif networkname == "PBR_CVPR2017_mod_concat":
        return networks.PBR_CVPR2017_mod_concat(out_size)
    elif networkname == "PBR_CVPR2017_mod_deep":
        return networks.PBR_CVPR2017_mod_deep(out_size)
    elif networkname == "FC_DenseNet67":
        return networks.FC_DenseNet67(out_size)
    else:
        return None
    

def modalSelect(task, modal, csvpath, bit, range_max, learn, reduce):
    if task == "normal_estimation":
        import modals.myModal_normEst as myModal
    elif task == "dif_spe_separation":
        import modals.myModal_difSpe as myModal
    elif task == "normal_disAmbiguate":
        import modals.myModal_disAmbig as myModal
        return myModal.Dataset_disAmbig(csvpath, bit, range_max, learn, reduce)
    elif task == "illumination_estimation":
        import modals.myModal_illumEst as myModal
        
    if modal == "s0s1s2":
        return myModal.Dataset_s0s1s2(csvpath, bit, range_max, learn, reduce)
    elif modal == "s0s1s2_gray":
        return myModal.Dataset_s0s1s2_gray(csvpath, bit, range_max, learn, reduce)
    elif modal == "only_s0":
        return myModal.Dataset_onlys0(csvpath, bit, range_max, learn, reduce)
    elif modal == "only_s0_gray":
        return myModal.Dataset_onlys0_gray(csvpath, bit, range_max, learn, reduce)
    elif modal == "only_s0_gray3ch":
        return myModal.Dataset_onlys0_gray3ch(csvpath, bit, range_max, learn, reduce)
    elif modal == "s0DopPhase":
        return myModal.Dataset_s0DopPhase(csvpath, bit, range_max, learn, reduce)
    elif modal == "s0DopPhase_gray":
        return myModal.Dataset_s0DopPhase_gray(csvpath, bit, range_max, learn, reduce)
    elif modal == "fourPolar":
        return myModal.Dataset_four_polar(csvpath, bit, range_max, learn, reduce)
    elif modal == "fourPolar_gray":
        return myModal.Dataset_four_polar_gray(csvpath, bit, range_max, learn, reduce)
    else:
        return None


def modalSelect_infer(task, modal, csvpath, bit, range_max, usenormal, learn, reduce):
    if task == "normal_estimation" and usenormal == False:
        import modals.myModal_normEst as myModal
    elif task == "dif_spe_separation" and usenormal == False:
        import modals.myModal_difSpe as myModal
    elif task == "normal_disAmbiguate":
        import modals.myModal_disAmbig as myModal
        return myModal.Dataset_disAmbig(csvpath, bit, range_max, usenormal, learn, reduce)
    elif task == "illumination_estimation":
        import modals.myModal_illumEst as myModal

    if modal == "s0s1s2":
        return myModal.Dataset_s0s1s2(csvpath, bit, range_max, learn, reduce)
    elif modal == "s0s1s2_gray":
        return myModal.Dataset_s0s1s2_gray(csvpath, bit, range_max, learn, reduce)
    elif modal == "only_s0":
        return myModal.Dataset_onlys0(csvpath, bit, range_max, learn, reduce)
    elif modal == "only_s0_gray":
        return myModal.Dataset_onlys0_gray(csvpath, bit, range_max, learn, reduce)
    elif modal == "only_s0_gray3ch":
        return myModal.Dataset_onlys0_gray3ch(csvpath, bit, range_max, learn, reduce)
    elif modal == "s0DopPhase":
        return myModal.Dataset_s0DopPhase(csvpath, bit, range_max, learn, reduce)
    elif modal == "s0DopPhase_gray":
        return myModal.Dataset_s0DopPhase_gray(csvpath, bit, range_max, learn, reduce)
    elif modal == "fourPolar":
        return myModal.Dataset_four_polar(csvpath, bit, range_max, learn, reduce)
    elif modal == "fourPolar_gray":
        return myModal.Dataset_four_polar_gray(csvpath, bit, range_max, learn, reduce)
    else:
        return None


def augmentationSelect(task, csvpath, bit, augname):
    if task == "normal_estimation":
        import augmentations.mydataAugmentation_normEst as myAug
    elif task == "dif_spe_separation":
        import augmentations.mydataAugmentation_difSpe as myAug
    elif task == "normal_disAmbiguate":
        import augmentations.mydataAugmentation_difSpe as myAug
    """elif task == "illumination_estimation":
        import augmentations.mydataAugmentation_normEst as myAug"""

    if augname == "flip_horizontal":
        return myAug.FlipImage_Horizontal(csvpath, bit, augname)
    elif augname =="blur":
        return myAug.Blur(csvpath, bit, augname, square=(3,3))
    elif augname == "noise":
        return myAug.Noise(csvpath, bit, augname, sigma=0.01)
    elif augname == "intensity":
        return myAug.Intensity(csvpath, bit, augname)
    else:
        return None
    

