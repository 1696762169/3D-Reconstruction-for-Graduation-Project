from torch.utils.data import Dataset
import argparse
import sys
import os
import numpy
import utils

def read_file_list(filepath):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    Origin: https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools
    """
    ret = { }
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(","," ").replace("\t"," ").strip()
            # 跳过注释行
            if len(line) == 0 or line[0] == '#':
                continue
            # 去除空格
            data = [value for value in line.split(" ") if value != '']
            # 添加到结果
            if len(data) > 1:
                ret[float(data[0])] = data[1:]
    return ret

def associate(first_dict: dict[float, list], second_dict: dict[float, list], offset, max_difference) -> dict[float, list]:
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    Origin: https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools
    """
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_dict 
                         for b in second_dict 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()

    first_set = set(first_dict.keys())
    second_set = set(second_dict.keys())
    matches = { }
    for _, a, b in potential_matches:
        if a in first_set and b in second_set:
            first_set.remove(a)
            second_set.remove(b)
            matches[a] = [str(value) for value in first_dict[a] + [b - offset] + second_dict[b]]

    return matches

def tum_preprocess(config):
    """进行TUM数据集的预处理"""
    tum_config = utils.DotDict(**config["tum"])

    rgb_dict = read_file_list(os.path.join(config.data_root, tum_config.rgb_file))
    depth_dict = read_file_list(os.path.join(config.data_root, tum_config.depth_file))
    gt_dict = read_file_list(os.path.join(config.data_root, tum_config.gt_file))
    offset = float(tum_config.match_offset)
    tolerance = float(tum_config.match_tolerance)

    # 合并数据
    matches = associate(rgb_dict, depth_dict, offset, tolerance)
    matches = associate(matches, gt_dict, offset, tolerance)

    with open(os.path.join(config.data_root, tum_config.associate_file), "w") as f:
        for key in matches.keys():
            print(matches[key])
            print(' '.join(matches[key]))
            f.write(f"{key} {' '.join(matches[key])}\n")

class TUMDataset(Dataset):
    pass