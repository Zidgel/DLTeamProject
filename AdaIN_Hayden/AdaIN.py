# Hayden Schennum
# 2025-04-19

import os
import random
import numpy as np
from PIL import Image
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]
random.seed(3)


class RandomPairDataset(Dataset):
    """
    interp. a dataset where each sample is content+style image pair:
    content image is sampled deterministically (sequentially) from given content dir;
    style image category is first sampled uniformly from style categories (in given style dir),
    then a style image is sampled uniformly from the chosen category
    - all images were pre-filtered to remove images with min(width,height) < 224
    RandomPairDataset has (in addition to Dataset attrs):
        content_img_paths is list<str> - paths of all content images
        style_cats is list<str> - name of each style category (each category is its own directory)
        style_cats_to_img_paths is dict<str:list<str>> - name of each style category, and paths of every corresponding style image
        transform is img<int>(W,H,3)->tens<float>(3,224,224) - series of operations to apply to each image before it is used as model input
    """
    def __init__(self, content_dir, style_dir):
        """
        RandomPairDataset str str -> RandomPairDataset
        given content dir and style dir, initializes a RandomPairDataset instance
        """
        self.content_img_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
        self.style_cats = [d for d in os.listdir(style_dir)]
        self.style_cats_to_img_paths = {} # dict<str:list<str>>
        for cat in self.style_cats:
            cat_path = os.path.join(style_dir, cat)
            self.style_cats_to_img_paths[cat] = [os.path.join(cat_path, f) for f in os.listdir(cat_path)]
        self.transform = transforms.Compose([
            transforms.Lambda(partial(resize_if_larger, thresh=448)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg_mean, std=vgg_std),
        ])


    def __len__(self):
        """
        RandomPairDataset -> int
        returns number of exs to use per epoch (i.e. number of content images)
        """
        return len(self.content_img_paths)
    

    def __getitem__(self, idx):
        """
        RandomPairDataset int -> tens<float>(3,224,224) tens<float>(3,224,224)
        given idx of this ex; returns corresponding (content,style) tensor pair
        """
        content_img_path = self.content_img_paths[idx]
        content_img = Image.open(content_img_path).convert("RGB") # img<int>(W,H,3)
        content_img_tens = self.transform(content_img) # tens<float>(3,224,224)

        style_cat = random.choice(self.style_cats)
        style_img_path = random.choice(self.style_cats_to_img_paths[style_cat])
        style_img = Image.open(style_img_path).convert("RGB") # img<int>(W,H,3)
        style_img_tens = self.transform(style_img) # tens<float>(3,224,224)

        return content_img_tens,style_img_tens



def resize_if_larger(img, thresh):
    """
    img<int>(W,H,3) int -> img<int>(W,H,3)
    if image's smallest dim is larger than given theshold,
    shrinks the image (maintaining aspect ratio) s.t. the smallest dim is equal to the threshold
    """
    w,h = img.size
    smallest_dim = min(w,h)
    if smallest_dim > thresh:
        resize_fx = transforms.Resize(thresh) # img<int>(W,H,3) -> img<int>(W,H,3)
        resized_img = resize_fx(img)
        return resized_img
    else: 
        return img
    





if __name__ == "__main__":
    N = 8 # batch size

    device = torch.device("cuda")
    test_dataset = RandomPairDataset("AdaIN_Hayden/data/content/test", "AdaIN_Hayden/data/style/test")
    # train_dataset = RandomPairDataset("AdaIN_Hayden/data/content/train", "AdaIN_Hayden/data/style/train")

    # one_ex = test_dataset[0]

    # test_loader = DataLoader(test_dataset, batch_size=N, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=N, shuffle=False, num_workers=1)

    for content_batch,style_batch in test_loader: # (N,3,224,224) and (N,3,224,224)
        y = 1

    x = 1



