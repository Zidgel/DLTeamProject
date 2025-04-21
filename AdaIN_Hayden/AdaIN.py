# Hayden Schennum
# 2025-04-19

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

class RandomPairDataset(Dataset):
    """
    interp. a dataset where each sample is content+style image pair:
    content image is sampled uniformly from given content dir;
    style image category is first sampled uniformly from style categories (in given style dir),
    then a style image is sampled uniformly from the chosen category
    - note: content images were pre-filtered to remove images with min(width,height) < 256
    RandomPairDataset has (in addition to Dataset attrs):
        content_img_paths is list<str> - paths of all possible content images
        style_cats_to_paths is dict<str:list<str>> - name of each style category, and paths of every corresponding style image

    """
    def __init__(self, content_dir, style_dir):
        """
        RandomPairDataset str str -> RandomPairDataset
        given content dir and style dir, initializes a RandomPairDataset instance
        """
        self.content_img_paths = []
        for f in os.listdir(content_dir):


        [os.path.join(content_dir, f) for f in ]

        style_cats_to_paths = {} # dict<str:list<str>>
        for cat in os.listdir(style_dir):
            cat_path = os.path.join(style_dir, cat)
            style_cats_to_paths[cat] = [os.path.join(cat_path, f) for f in os.listdir(cat_path)]





if __name__ == "__main__":
    N = 8 # batch size

    device = torch.device("cuda")
    dataset = RandomPairDataset("AdaIN_Hayden/data/content", "AdaIN_Hayden/data/style")



