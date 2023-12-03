from torch.utils.data import Dataset
import torch
import os
import numpy as np
import cv2
from math import sqrt
from tqdm import tqdm
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
    USE_RGB = False

    def __init__(self, config: utils.DotDict, online: bool = False) -> None:
        """TUM数据集
        :param config: 配置文件
        :param online: 是否在使用时才读取数据"""
        super().__init__()
        self.config = config
        # 读取关联文件内容 避免反复打开文件
        associate_file_path = os.path.join(self.config.data_root, self.config.tum.associate_file)
        with open(associate_file_path, "r") as af:
            self.associate = af.readlines()

        # 内参矩阵
        self.K = utils.get_calib_matrix(self.config.data_type)
        # 读取数据
        self.data = [None] * len(self.associate)
        if not online:
            print("Loading data from", self.config.data_root, "...")
            for index, line in enumerate(tqdm(self.associate)):
                self.__load_data(line, index)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.data[index] is None:
            self.data[index] = self.__load_data(self.associate[index], index)
        return self.data[index]
    
    def __load_data(self, associate: str, index: int):
        """读取一行数据"""
        data = associate.strip().split(" ")
        root = self.config.data_root
        # 深度图像
        depth = cv2.imread(os.path.join(root, data[3]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / self.config.tum.depth_scale
        invalid = (depth < self.config.depth_near) | (depth > self.config.depth_far)
        depth[invalid] = -1
        # RGB图像
        rgb = cv2.imread(os.path.join(root, data[1])) if TUMDataset.USE_RGB else None
        # 位姿
        pose = TUMDataset.__tum2homogeneous([float(value) for value in data[5:]])
        
        self.data[index] = TUMData(torch.from_numpy(depth), torch.from_numpy(rgb) if rgb else None, torch.from_numpy(pose))
    
    @staticmethod
    def __tum2homogeneous(gt_pose: list[float]) -> np.ndarray:
        """将TUM数据集的位姿转换为齐次矩阵形式"""
        # 解析位移信息
        ret = np.eye(4)
        ret[:3, 3] = gt_pose[:3]
        t = gt_pose[:3]

        # 解析旋转信息
        q = np.array(gt_pose[3:], dtype=np.float64, copy=True)
        norm = np.linalg.norm(q)
        # 判断四元数模长是否过小 并进行归一化 为后续计算方便 归一化结果的模长为sqrt(2.0)
        if norm < np.finfo(np.float64).eps:
            return ret
        q /= norm / sqrt(2.0)

        # 将四元数转换为旋转矩阵 此处四元数的顺序是x, y, z, w
        q = np.outer(q, q)
        ret[:3, :3] = np.array([
            [1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]],
            [    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]],
            [    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1]]])
        return ret

class TUMData(object):
    def __init__(self, depth, rgb, pose) -> None:
        self.depth = depth
        self.rgb = rgb
        self.pose = pose