# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:45:21 2018

@author: 0000145046
"""

import pathlib

import cv2
import numpy as np
import yaml

from chainer import serializers
import chainer
from chainer.cuda import to_cpu

import mymodules.myclass.myClassLoader as myClassLoader
from mymodules.myutils.pathutils import loadParamsFromConditiontxt
from mymodules.myutils.pathutils import parse_filePath
from mymodules.myutils.pathutils import extractID
import mymodules.myutils.polarutils as pol


#--------------------------
# Load parameter_train.yml
#--------------------------
with open("./parameters_learn.yml") as f:
    infparams = yaml.load(f)["inference"]

task = infparams["task"]
result = infparams["result"]
usenormal = infparams["usenormal"]
csvpaths = infparams["csvpaths"]
epoch_nums = infparams["epoch_num"]
bit = infparams["bit"]

filepath = pathlib.Path("./{}/Result/{}/".format(task, result))

lnparams = loadParamsFromConditiontxt(
        filepath.joinpath("condition.txt"))
modal       = lnparams["modal"]
networkname = lnparams["networkname"]
range_max   = float(lnparams["range_max"])


for epoch_num in epoch_nums:
    # Load learned model
    #--------------------------
    net = myClassLoader.networkSelect(networkname, task)
    serializers.load_npz(
            filepath.joinpath("Model", "snapshot_epoch-{:d}".format(epoch_num)),
            net)
    
    # Load test dataset
    #--------------------------
    for csvpath in csvpaths:
        test = myClassLoader.modalSelect_infer(
                task, modal, [pathlib.Path(csvpath)], bit, range_max,
                usenormal, False, reduce=1)
        
        test_stokes = []
        savenames   = []
        for imgnum in range(len(test)):
            test_stokes.append(test.get_example(imgnum))
            
            s0Path = pathlib.Path(test.pairs[imgnum, 0])
            savenames.append(extractID(s0Path.name))
        
        # Inference
        #--------------------
        gpu_id = 0
        
        if gpu_id >= 0:
            net.to_gpu(gpu_id)
        
        x = net.xp.asarray(test_stokes)
        
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            
            if "DenseNet" in networkname: # DenseNet is exception because of GPU memory capacity.
                imgnum,h,w = x.shape[0], x.shape[2], x.shape[3]
                if task == "normal_estimation" or task == "normal_disAmbiguate":
                    y = net.xp.zeros((imgnum,3,h,w), dtype=net.xp.float32)
                elif task == "dif_spe_separation":
                    y = net.xp.zeros((imgnum,6,h,w), dtype=net.xp.float32)
                for num in range(x.shape[0]):
                    y[num:(num+1),:,:,:] += net(x[num:(num+1),:,:,:]).array
            else:
                y = net(x)
                y = y.array 
        
        # Save
        #--------------------
        
        # Make save directory
        picfolder = parse_filePath(csvpath, 2, down=True)
        savepath = filepath.joinpath("Inference", "epoch-{:d}".format(epoch_num), picfolder)
        savepath.mkdir(parents=True, exist_ok=True)
        
        if task == "dif_spe_separation":
            savepath_normal = savepath.joinpath("normal")
            savepath_normal.mkdir(exist_ok=True)
        
        for testnum in range(y.shape[0]):
            result = y[testnum].transpose(1, 2, 0)
            
            if task == "normal_estimation" or task == "normal_disAmbiguate":
                img = result[:, :, ::-1]
                img = to_cpu(img)
                img = np.clip(img, 0, range_max)
                img = img * 255 / range_max
                #img = np.clip(img, -1, 1)
                #img = ((img + 1)/2) * 255.
                
                savename = savenames[testnum]
                cv2.imwrite(
                        str(savepath.joinpath("{}.png".format(savename))),
                        img.astype(np.uint8))
            
            elif task == "dif_spe_separation":
                result_names = ["s0Spe","DopSpe","PhaseSpe","s0Dif","DopDif","PhaseDif"]
                for resultnum, result_name in enumerate(result_names):
                    img = result[:,:,resultnum]
                    img = to_cpu(img)
                    img = np.clip(img, 0, range_max)
                    img = img * 65535 / range_max
                    
                    savename = savenames[testnum]
                    cv2.imwrite(
                            str(savepath.joinpath("{}_{}.png".format(savename, result_name))),
                            img.astype(np.uint16))
                    
                # make normal from reconstructed DoP and Phase
                dop = cv2.imread(
                        str(savepath.joinpath("{}_DopDif.png".format(savename))),
                        -1).astype(np.float32)
                
                phase = cv2.imread(
                        str(savepath.joinpath("{}_PhaseDif.png".format(savename))),
                                 -1).astype(np.float32)
                
                dop = dop / 65535.
                phase = phase * 180. / 65535.
                
                normal = pol.calc_normal_from_dop_and_phase(dop, phase)
                cv2.imwrite(
                        str(savepath_normal.joinpath("{}.png".format(savename))),
                        normal.astype(np.uint8))

