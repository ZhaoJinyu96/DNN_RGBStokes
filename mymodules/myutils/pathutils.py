# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:37:31 2019

@author: 0000145046
"""

import pathlib

def find_all_csvpath_recursively(directories):
    if not isinstance(directories, list):
        raise TypeError("directories must be list object of str")
        
    else:
        csvpaths = []
        
        for directory in directories:
            path = pathlib.Path(directory)
            csvpaths.extend([p.parent for p in path.glob("**/*.csv")])
            
    return csvpaths

def parse_filePath(path, num, down=False):
    
    path = pathlib.Path(path)
    blockNum = len(path.parents)
    
    if num < 1:
        raise ValueError("num:{} should be larger than 0".format(num))
    elif num > blockNum:
        raise ValueError("num:{} should be smaller than {}".format(num, blockNum))
        
    blockList = []
    while blockNum > 0:
        blockList.append(path.name)
        path = path.parent
        blockNum -= 1
        
    if not down:
        blockList = blockList[::-1]
        result = blockList[0]
        for i in range(0, num-1):
            result = "{}/{}".format(result, blockList[i+1])
    else:
        result = blockList[0]
        for i in range(0, num-1):
            result = "{}/{}".format(blockList[i+1], result)
    
    return result


def extractID(imgName):
    """
    extract "ID-0" from input image name.
    In:
        str, name of image.
        The name format should follow the following example.
        "ID-0_hoge_hoge_hoge_++++++.png"
    
    Out:
        "ID-0", str.
    """
    idn = imgName.replace(".png", "").split("_")[0]
    
    if "ID-" in idn:
        return idn
    else:
        raise ValueError("Cannot extract ID")


def find_same_id(gtname, recons):
    gtid = extractID(gtname.name)
    recons = list(recons)
    for recon in recons:
        reconid = extractID(recon.name)
        if gtid == reconid:
            return recon
    
    raise FileNotFoundError("'gt' and 'recon' do not have the same id.")
    
    
def find_same_id_fromKey(gtname, recons, key):
    gtid = extractID(gtname.name)
    
    for recon in recons:
        if key in recon.name:
            reconid = extractID(recon.name)
            if gtid == reconid:
                return recon
    return 0


def find_same_id_s0(gtname, recons):
    recons = list(recons)
    tar = find_same_id_fromKey(gtname, recons, "s0")
    if tar == 0:
        tar = find_same_id_fromKey(gtname, recons, "S0")
    if tar == 0:
        raise FileNotFoundError("'s0' and 'recon' do not have the same id.")
        
    return tar


def find_same_id_s1(gtname, recons):
    recons = list(recons)
    tar = find_same_id_fromKey(gtname, recons, "s1")
    if tar == 0:
        tar = find_same_id_fromKey(gtname, recons, "S1")
    if tar == 0:
        raise FileNotFoundError("'s1' and 'recon' do not have the same id.")
        
    return tar


def find_same_id_s2(gtname, recons):
    recons = list(recons)
    tar = find_same_id_fromKey(gtname, recons, "s2")
    if tar == 0:
        tar = find_same_id_fromKey(gtname, recons, "S2")
    if tar == 0:
        raise FileNotFoundError("'s2' and 'recon' do not have the same id.")
        
    return tar


def loadParamsFromConditiontxt(name):
    f = open(name)
    lines = f.readlines()
    
    params = {}
    cnt = 0
    while lines[cnt][0] != "-":
        line = lines[cnt].replace(" ", "").replace("\n", "").split(":")
        params[line[0]] = line[1]
        
        cnt += 1
    
    f.close()
    return params


def calcDesignatedPathBlock(path, num, down=False):
    if num <= 1:
        raise ValueError("num should be larger than 2.")
    
    a = parse_filePath(path, num-1, down)
    b = parse_filePath(path, num, down)
    
    if b.count(a) > 1:
        raise ValueError("Input path includes several blocks which have the same name.")
    
    return b.replace(a, "").replace("/", "")


def returnSortedEpochNum(name_list):
    epoch_nums = []
    for name in name_list:
        epoch_nums.append(int(name.split("-")[-1]))
    epoch_nums.sort()
    
    return epoch_nums     


def returnGTPaths(task, folderpart):
    if task == "normal_estimation":
        paths = pathlib.Path(
            "../{}/Picture/{}/gt/".format(task, folderpart)).glob("*.png")

    elif task == "dif_spe_separation":
        paths = pathlib.Path(
            "../{}/Picture/{}/gt_amb/".format(task, folderpart)).glob("*.png")
        
    elif task == "normal_disAmbiguate":
        paths = pathlib.Path(
            "../dif_spe_separation/Picture/{}/gt/".format(folderpart)).glob("*.png")
        
    return list(paths)

def returnMaskPaths(task, folderpart):
    if task == "normal_disAmbiguate":
        task = "dif_spe_separation"
    
    return list(pathlib.Path(
            "../{}/Picture/{}/mask/".format(task, folderpart)).glob("*.png"))

def returnS0Paths(task, folderpart):
    if task == "normal_disAmbiguate":
        task = "dif_spe_separation"
        
    return list(pathlib.Path(
            "../{}/Picture/{}/train/".format(task, folderpart)).glob("*.png"))

def returnRecPaths(task, resultname, epochnum, folderpart):
    if task == "normal_estimation" or task == "normal_disAmbiguate":
        recpaths = pathlib.Path(
                "../{}/Result/{}/Inference/epoch-{}/{}/".format(
                    task, resultname, epochnum, folderpart)).glob("*.png")

    elif task == "dif_spe_separation":
        recpaths = pathlib.Path(
                "../{}/Result/{}/Inference/epoch-{}/{}/normal/".format(
                    task, resultname, epochnum, folderpart)).glob("*.png")

    return list(recpaths)


def trainPath2spePath(path):
    trainDir = pathlib.Path(path).name
    return path.joinpath("gt_difspe", "{}_spe".format(trainDir))

def trainPath2difPath(path):
    trainDir = pathlib.Path(path).name
    return path.joinpath("gt_difspe", "{}_dif".format(trainDir))
    
    
