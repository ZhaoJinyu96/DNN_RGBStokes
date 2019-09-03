# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:53:23 2018

@author: 0000145046
"""

import yaml
import pathlib

import numpy as np

import mymodules.myclass.myClassLoader as myClassLoader
from mymodules.myutils.pathutils import find_all_csvpath_recursively

import chainer
from chainer import iterators
from chainer.datasets import split_dataset_random
from chainer.cuda import to_cpu
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer import serializers


#--------------------------
# Load parameter_train.yml
#--------------------------
with open("./parameters_learn.yml") as f:
    params = yaml.load(f)["train"]


task          = params["task"]
modal         = params["modal"]
networkname   = params["networkname"]
batchsize     = params["batchsize"]
bit           = params["bit"]
range_max     = params["range_max"]
max_epoch     = params["max_epoch"]
reduce        = params["reduce"]
lossname      = params["lossname"]
optimizername = params["optimizername"]
lr            = params["lr"]
wd            = params["wd"]
savename      = params["savename"] + "_{}_{}".format(modal, networkname)
gpu_id        = params["gpu_id"]
csvpath       = find_all_csvpath_recursively(params["csvpath"])
sequence_num  = params["sequence_num"]

#--------------------------
# Define dataset
#--------------------------
train_dataset = myClassLoader.modalSelect(task, modal, csvpath, bit, range_max, True, reduce)

train, valid = split_dataset_random(train_dataset, int(len(train_dataset)*0.8), seed=0)
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

#--------------------------
# Create condition memo
#--------------------------
savepath = pathlib.Path("./{}/Result/{}".format(task, savename))

if sequence_num == 0: # learn from initial epoch
    savepath.mkdir()
    f = open(savepath.joinpath("condition.txt"), "w")
    
    f.write("modal             :" + modal           + "\n")
    f.write("networkname       :" + networkname     + "\n")
    f.write("batchsize         :" + str(batchsize)  + "\n")
    f.write("bit               :" + str(bit)        + "\n")
    f.write("range_max         :" + str(range_max)  + "\n")
    f.write("training data num :" + str(len(train)) + "\n")
    f.write("lossfun           :" + str(lossname)   + "\n")
    f.write("optimizer name    :" + optimizername   + "\n")
    f.write("learning rate     :" + str(lr)         + "\n")
    f.write("weight decay      :" + str(wd)         + "\n")
    
    f.write("--------------------------------------------------------\n")
    
    for csvnum in range(len(csvpath)):
        f.write("csvpath%03d: " %csvnum + str(csvpath[csvnum]) + "\n")
    
    f.close()


#--------------------------
# Define network
#--------------------------
net = myClassLoader.networkSelect(networkname, task)


if sequence_num > 0: # learn from interrupted epoch
    # Load learned parameters
    serializers.load_npz(
            savepath.joinpath("Model/snapshot_epoch-{:d}".format(sequence_num)),
            net)

if gpu_id >= 0:
    net.to_gpu(gpu_id)

# Set loss function
lossfun = myClassLoader.lossFunselect(lossname)
net.compute_accuracy=False

# Set the optimizer
if optimizername == "Adam":
    optimizer = optimizers.Adam(alpha=lr, weight_decay_rate=wd)
elif optimizername == "Amsgrad":
    optimizer = optimizers.Adam(alpha=lr, weight_decay_rate=wd, amsgrad=True)
elif optimizername == "SGD":
    optimizer = optimizers.SGD(lr=lr)
    optimizer.add_hook(chainer.optimizer.WeightDecay(wd))
elif optimizername == "RMSprop":
    optimizer = optimizers.RMSprop(lr=lr)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(wd))

optimizer.setup(net)

if sequence_num > 0:
    # Load optimizer
    serializers.load_npz(savepath.joinpath("my.state"), optimizer)

#--------------------------
# Main iteration
#--------------------------
if sequence_num == 0:
    savepath.joinpath("Model").mkdir()

while train_iter.epoch < max_epoch:
    # Output loss value in a txt file
    f = open(savepath.joinpath("loss.txt"), "a")
    
    train_batch = train_iter.next()
    x, t, mask = concat_examples(train_batch, gpu_id)
    
    # Prediction
    y = net(x)
    loss = lossfun(y, t, mask)
    
    net.cleargrads()
    loss.backward()
    
    optimizer.update()
    
    epochnum = train_iter.epoch + sequence_num
    # Check
    if train_iter.is_new_epoch:
        # Print loss value
        f.write('epoch:{:04d}  train_loss:{:.04f} '.format(
                epochnum, float(to_cpu(loss.data))))
        
        valid_losses = []
        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid, mask_valid = concat_examples(valid_batch, gpu_id)
            
            with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
                
                    y_valid = net(x_valid)
            
            loss_valid = lossfun(y_valid, t_valid, mask_valid)
            valid_losses.append(to_cpu(loss_valid.array))
            
            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break
                
        f.write('  val_loss:{:.04f}\n'.format(np.mean(valid_losses)))
        
        # Save model
        if epochnum % 10 == 0: # Every ten epoch
            serializers.save_npz(
                    str(savepath.joinpath("Model/snapshot_epoch-{:d}".format(epochnum))),
                    net)
            serializers.save_npz(
                    str(savepath.joinpath("my.state")),
                    optimizer)

    f.close()
