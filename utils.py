import time
import torch
import numpy as np
import yaml

class DotDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self[key] = DotDict(**value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f"'ForceKeyErrorDict' object has no attribute '{item}'") from e
    def __setattr__(self, key, value):
        self[key] = value
    def __missing__(self, key):
        raise KeyError(key)

def load_yaml(path):
    r"""加载yaml文件"""
    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = DotDict(**config_dict)

    return config

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

def get_calib(data_type: str):
    r"""
    获取相机校准内参
    数据来源: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
    """
    if data_type == "fr1":
        return {
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3
        }
    elif data_type == "fr2":
        return {
            "fx": 520.9,
            "fy": 521.0,
            "cx": 325.1,
            "cy": 249.7
        }
    elif data_type == "fr3":
        return {
            "fx": 535.4,
            "fy": 539.2,
            "cx": 320.1,
            "cy": 247.6
        }

def get_calib_matrix(data_type: str):
    r"""
    获取相机校准内参矩阵
    数据来源: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
    """
    calib_list = get_calib(data_type)
    return np.array([
        [calib_list["fx"], 0, calib_list["cx"]],
        [0, calib_list["fy"], calib_list["cy"]],
        [0, 0, 1]
    ]) if calib_list else None

def bilateral_filter(image, diameter: float, sigma_i: float, sigma_s: float):
    """
    对图像应用双边滤波。
    """
    sigma_i = 2 * sigma_i ** 2
    sigma_s = 2 * sigma_s ** 2

    def apply_bilateral_filter(x: int, y: int):
        """
        对单个像素应用双边滤波。
        """
        hl = diameter // 2
        i_filtered = 0
        Wp = 0
        for k in range(max(x - hl, 0), min(x + hl + 1, image.shape[0])):
            for l in range(max(y - hl, 0), min(y + hl + 1, image.shape[1])):
                gi = np.exp(-((image[k][l] - image[x][y]) ** 2 / sigma_i))
                gs = np.exp(-((k - x) ** 2 + (l - y) ** 2 / sigma_s))
                w = gi * gs
                i_filtered += image[k][l] * w
                Wp += w
        return i_filtered / Wp
    
    new_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = apply_bilateral_filter(i, j)
        if i % 10 == 0:
            print(f"Progress: {i}/{image.shape[0]}")
    return new_image