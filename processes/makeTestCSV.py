# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:54:06 2018

@author: 0000145046
"""

import yaml
import pathlib

import numpy as np


with open("./parameters_process.yml") as f:
    params = yaml.load(f)["makeTestCSV"]

path = pathlib.Path(params["path"])

s0_names = list(path.joinpath("train").glob("*S0.png"))

csv = np.empty((0, 3), dtype=str)
for s0_name in s0_names:
    s0_name = str(s0_name.name)
    
    if s0_name.find("S0") == -1:
        raise ValueError("S0 should be a chapital letter.")
    if s0_name.find("ID-") == -1:
        raise ValueError("Input image names should be started with 'ID-'")
        
    s1_name = s0_name.replace("S0", "S1")
    s2_name = s0_name.replace("S0", "S2")
    
    csv = np.append(csv,
        np.array([[s0_name, s1_name, s2_name]]), axis=0)


np.savetxt(path.joinpath("path.csv"), csv, delimiter=",", fmt="%s")
