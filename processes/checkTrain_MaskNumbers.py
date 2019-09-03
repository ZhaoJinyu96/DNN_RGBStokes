# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:23:02 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import yaml

from mymodules.myutils.pathutils import find_all_csvpath_recursively


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["checkTrain_MaskNumbers"]
    
folders = find_all_csvpath_recursively(params["path"])

for folder in folders:
    gtimgs = list(folder.joinpath("gt").iterdir())
    maskimgs = list(folder.joinpath("mask").iterdir())
    
    if len(gtimgs) == len(maskimgs):
        pass
    else:
        print(folder)

