# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:13:23 2018

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml

from mymodules.myclass import myClassLoader
from mymodules.myutils.pathutils import find_all_csvpath_recursively


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["dataAugmentation"]

task = params["task"]
csvpath = find_all_csvpath_recursively(params["csvpath"])
augnames = params["augnames"]
bit = params["bit"]


for augname in augnames:
    data = myClassLoader.augmentationSelect(task, csvpath, bit, augname)  
    data.run_augmentation_and_save()
