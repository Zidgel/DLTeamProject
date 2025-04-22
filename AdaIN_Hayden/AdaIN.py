# Hayden Schennum
# 2025-04-19

import os
import random
import numpy as np
from PIL import Image
from functools import partial
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(3)


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
        transform is img<int>(W,H,3)->tens<float>(3,224,224) - series of operations to apply to each image before it is used as model input (training)
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


class Encoder(nn.Module):
    """
    interp. the (frozen) encoder component of an STN
    Encoder has (in addition to nn.Module attrs):
        layers is nn.Sequential - partial sequence of frozen layers from VGG-19
    """
    def __init__(self):
        """
        Encoder -> Encoder
        initializes an Encoder instance
        """
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features # nn.Sequential
        self.layers = vgg19[:21].eval() # all layers up to and including relu4_1
        for param in self.layers.parameters():
            param.requires_grad = False
    
    def forward(self,x):
        """
        Encoder tens<float>(N,3,H,W) -> tens<float>(N,512,H/8,W/8)
        given input image; returns extracted features from image
        """
        out = self.layers(x) # (N,512,H/8,W/8)
        return out
    

class AdaIN(nn.Module):
    """
    interp. the (no params) AdaIN component of an STN
    AdaIN has (in addition to nn.Module attrs): n/a
    """
    def __init__(self):
        """
        AdaIN -> AdaIN
        initializes an AdaIN instance
        """
        super().__init__()
    
    def forward(self,c,s):
        """
        AdaIN tens<float>(N,512,Hc/8,Wc/8) tens<float>(N,512,Hs/8,Ws/8) -> tens<float>(N,512,Hc/8,Wc/8)
        given content feats and style feats;
        returns version of content feats s.t. mean and var match those of style feats
        - means/stds are computed on a per-channel, per-ex basis
        """
        c_mean = c.mean([2,3], keepdim=True) # (N,512,1,1)
        c_std = c.std([2,3], keepdim=True) + 1e-12 # (N,512,1,1)
        s_mean = s.mean([2,3], keepdim=True) # (N,512,1,1)
        s_std = s.std([2,3], keepdim=True) # (N,512,1,1)
        adain_out = s_std*(c-c_mean)/c_std + s_mean # (N,512,Hc/8,Wc/8)
        return adain_out


class Decoder(nn.Module):
    """
    interp. the (learned) decoder component of a STN
    Decoder has (in addition to nn.Module attrs):
        layers is nn.Sequential - sequence of layers that mirror the encoder component
    """
    def __init__(self):
        """
        Decoder -> Decoder
        initializes a Decoder instance (layer hparams are hard-coded)
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,256,H/8,W/8)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # (N,256,H/4,W/4)

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,256,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,256,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,256,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,128,H/4,W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # (N,128,H/2,W/2)

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,128,H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,64,H/2,W/2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # (N,64,H,W)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,64,H,W)
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), # (N,3,H,W)
        )

    def forward(self,x):
        """
        Decoder tens<float>(N,512,H/8,W/8) -> tens<float>(N,3,H,W)
        given AdaIN-normalized content feats; returns generated image
        """
        out = self.layers(x) # (N,3,H,W)
        return out        


class StyleTransferNet(nn.Module):
    """
    interp. the Style Transfer Net (STN) of the AdaIN architecture, used during inference
    StyleTransferNet has (in addition to nn.Module attrs):
        enc is Encoder - encoder component
        adain is AdaIN - adaptive instance normalization component
        dec is Decoder - decoder component
        alpha is float[0,1] - style transfer focus
    """
    def __init__(self,enc,adain,dec,alpha=1):
        """
        StyleTransferNet Encoder AdaIN Decoder -> StyleTransferNet
        initializes a StyleTransferNet instance
        """
        super().__init__()
        self.enc = enc
        self.adain = adain
        self.dec = dec
        self.alpha = alpha
    
    def forward(self,c,s):
        """
        StyleTransferNet tens<float>(N,3,Hc,Wc) tens<float>(N,3,Hs,Ws) -> tens<float>(N,3,Hc,Wc)
        given content img and style img; returns the generated img AND AdaIN tgt feats
        """
        c_feats = self.enc(c) # (N,512,Hc/8,Wc/8)
        s_feats = self.enc(s) # (N,512,Hs/8,Ws/8)
        adain_out = self.adain(c_feats,s_feats) # (N,512,Hc/8,Wc/8)
        gen = self.dec((1-self.alpha)*c_feats + self.alpha*adain_out) # (N,3,Hc,Wc)
        return gen, adain_out


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
    

def norm_tens_to_denorm_img(tens):
    """
    tens<float>(3,H,W) -> img<int>(W,H,3)
    given normalized tensor; returns denormalized PIL image
    """
    vgg_mean_tens = torch.tensor(vgg_mean).view(-1,1,1) # (3,1,1)
    vgg_std_tens = torch.tensor(vgg_std).view(-1,1,1) # (3,1,1)
    denorm_tens = tens*vgg_std_tens + vgg_mean_tens # (3,H,W)
    clamped_tens = torch.clamp(denorm_tens, 0, 1)
    to_pil = transforms.ToPILImage() # tens<float>(3,H,W) -> img<int>(W,H,3)
    pil_img = to_pil(clamped_tens) # img<int>(W,H,3)
    return pil_img




if __name__ == "__main__":
    batch_size = 8
    num_epochs = 1
    weight_decay = 1e-3
    lr = 1e-4 * batch_size
    lbda = 10 # style weight
    num_workers = 4
    device = torch.device("cuda")

    train_dataset = RandomPairDataset("AdaIN_Hayden/data/content/train", "AdaIN_Hayden/data/style/train")
    val_dataset = RandomPairDataset("AdaIN_Hayden/data/content/val", "AdaIN_Hayden/data/style/val")
    test_dataset = RandomPairDataset("AdaIN_Hayden/data/content/test", "AdaIN_Hayden/data/style/test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    

    for epoch in range(0,num_epochs):
        for batch_id,(content_batch,style_batch) in enumerate(train_loader): # (N,3,224,224) and (N,3,224,224)
            
            y = 1


    x = 1



