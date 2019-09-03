# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:48:01 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml
import numpy as np
import subprocess

from chainer import Variable
import chainer.computational_graph as c

from modules.myClassLoader import networkSelect


# Define parameters
with open("./parameters_visualize.yml") as f:
    params = yaml.load(f)["visualize_network"]

networkname = params["networkname"]
task    = params["task"]
in_size     = params["in_size"]
batch_size  = params["batch_size"]
height      = params["height"]
width       = params["width"]

# Load network
net = networkSelect(networkname, task)

# generate virtual dataset
x = np.empty((
        batch_size,
        in_size,
        height,
        width), dtype=np.float32)
x = Variable(x)

# input network
y = net(x)

# output graph
g = c.build_computational_graph(y)
with open(networkname+".dot", "w") as o:
    o.write(g.dump())
    
# convert .dot into .png
cmd = r"dot -Tpng {}.dot -o {}.png".format(networkname, networkname)
returncode = subprocess.Popen(cmd, shell=True)
