# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:47:47 2019

@author: 0000145046
"""

import sys
sys.path.append("../")

import shutil
import yaml

from mymodules.myutils.pathutils import find_all_csvpath_recursively


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["deleteFolders"]

path = params["path"]
deletename = params["deletename"]

csvpaths = find_all_csvpath_recursively(path)

for csvpath in csvpaths:
    if deletename in str(csvpath):
        shutil.rmtree(csvpath)
