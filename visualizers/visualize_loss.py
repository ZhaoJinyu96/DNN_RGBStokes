# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:38:56 2019

@author: 0000145046
"""

import sys
sys.path.append("../modules/")

import yaml
import matplotlib.pyplot as plt

import my_matplotlib_funcs as myplt

#----------------------
# define file names
#----------------------
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["visualize_loss"]
    
path = params["path"]


#----------------------
# define functions
#----------------------
def my_split(line):
    line = line.split(" ")
    new_line = [i for i in line if len(i) > 0]
    
    return new_line

#----------------------
# load loss.txt
#----------------------
train_loss = []
test_loss  = []

f = open(path + "loss.txt", "r")
lines = f.readlines()
f.close()

for line in lines:
    line = my_split(line)
    
    train_loss.append(float(line[1].split(":")[1]))
    test_loss.append(float(line[2].split(":")[1]))


#----------------------
# plot
#----------------------
fig = plt.figure()
ax = fig.add_subplot(111)

x = range(1, len(train_loss)+1)

ax.plot(x, test_loss, label="test")
ax.plot(x, train_loss, label="train")

ax.set_yscale("log")
ax.grid(which="both")
ax.set_title("loss on each epoch", fontsize=18)
ax.set_ylim(10**0, 10**3)
ax.set_xlabel("epoch number", fontsize=15)
ax.set_ylabel("value of loss", fontsize=15)
ax.legend(fontsize=12)

fig.show()
myplt.my_savefig(fig, path+"result.png")