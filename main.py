import os
from argparse import ArgumentParser
import utils
import dataset
import cv2
import torch
import numpy as np

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file.")
    args = parser.parse_args()

    # 获取配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError("config file not found.")
    config = utils.load_yaml(args.config)

    # 进行预处理
    data_root = config["data_root"]
    associate_file = os.path.join(data_root, config["tum"]["associate_file"])
    if not os.path.exists(associate_file):
        dataset.tum_preprocess(config)

    dataset = dataset.TUMDataset(config)

    # with open(associate_file, "r") as af:
    #     for line in af.readlines():
    #         _, rgb, _, depth = line.strip().split(" ")
    #         rgb = os.path.join(data_root, rgb)
    #         depth = os.path.join(data_root, depth)
    #         rgb = cv2.imread(rgb)
    #         depth = cv2.imread(depth)
    #         cv2.imshow("rgb", rgb)
    #         rgb1 = cv2.bilateralFilter(rgb, 3, 50, 50)
    #         cv2.imshow("rgb1", rgb1)
    #         rgb2 = cv2.bilateralFilter(rgb, 5, 100, 100)
    #         cv2.imshow("rgb2", rgb2)
    #         # cv2.imshow("depth", depth)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #         break
