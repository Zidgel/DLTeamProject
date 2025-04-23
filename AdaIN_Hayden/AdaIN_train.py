# Hayden Schennum
# 2025-04-19

import os, random, math, time
import numpy as np
from PIL import Image
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(3)
torch.use_deterministic_algorithms(True)


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
        layers1 is nn.Sequential - frozen layers from VGG-19, up to relu1_1
        layers2 is nn.Sequential - frozen layers from VGG-19, up to relu2_1
        layers3 is nn.Sequential - frozen layers from VGG-19, up to relu3_1
        layers4 is nn.Sequential - frozen layers from VGG-19, up to relu4_1
    """
    def __init__(self):
        """
        Encoder -> Encoder
        initializes an Encoder instance
        """
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features # nn.Sequential
        self.layers1 = vgg19[:2] # up to relu1_1
        self.layers2 = vgg19[2:7] # up to relu2_1
        self.layers3 = vgg19[7:12] # up to relu3_1
        self.layers4 = vgg19[12:21] # up to relu4_1
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self,x):
        """
        Encoder tens<float>(N,3,H,W) -> tens<float>(N,64,H,W) tens<float>(N,128,H/2,W/2) tens<float>(N,256,H/4,W/4) tens<float>(N,512,H/8,W/8)
        given input image; returns extracted features from image at each of the 4 chosen layers
        """
        relu1_1 = self.layers1(x) # (N,64,H,W)
        relu2_1 = self.layers2(relu1_1) # (N,128,H/2,W/2)
        relu3_1 = self.layers3(relu2_1) # (N,256,H/4,W/4)
        relu4_1 = self.layers4(relu3_1) # (N,512,H/8,W/8)
        return relu1_1,relu2_1,relu3_1,relu4_1
    

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
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,256,H/8,W/8)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # (N,256,H/4,W/4)

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,256,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,256,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,256,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,128,H/4,W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # (N,128,H/2,W/2)

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,128,H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,64,H/2,W/2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # (N,64,H,W)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,64,H,W)
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode="zeros"), # (N,3,H,W)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.layers[-1]:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        """
        Decoder tens<float>(N,512,H/8,W/8) -> tens<float>(N,3,H,W)
        given AdaIN-normalized content feats; returns generated image
        """
        out = self.layers(x) # (N,3,H,W)
        return out        


class StyleTransferNet(nn.Module):
    """
    interp. the Style Transfer Net (STN) of the AdaIN architecture
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
        StyleTransferNet tens<float>(N,3,Hc,Wc) tens<float>(N,3,Hs,Ws) 
        -> tens<float>(N,3,Hc,Wc) tens<float>(N,512,Hc/8,Wc/8) tens<float>(N,64,Hs,Ws) tens<float>(N,128,Hs/2,Ws/2) tens<float>(N,256,Hs/4,Ws/4) tens<float>(N,512,Hs/8,Ws/8)
        given content img and style img; returns the generated img AND AdaIN output feats (t) AND style feats at each chosen layer
        """
        _,_,_,c_41 = self.enc(c) # (N,512,Hc/8,Wc/8)
        s_11,s_21,s_31,s_41 = self.enc(s) # (N,64,Hs,Ws) (N,128,Hs/2,Ws/2) (N,256,Hs/4,Ws/4) (N,512,Hs/8,Ws/8)
        t = self.adain(c_41,s_41) # (N,512,Hc/8,Wc/8)
        gen = self.dec((1-self.alpha)*c_41 + self.alpha*t) # (N,3,Hc,Wc)
        return gen,t,s_11,s_21,s_31,s_41


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
    device = tens.device
    vgg_mean_tens = torch.tensor(vgg_mean, device=device).view(-1,1,1) # (3,1,1)
    vgg_std_tens = torch.tensor(vgg_std, device=device).view(-1,1,1) # (3,1,1)
    denorm_tens = tens*vgg_std_tens + vgg_mean_tens # (3,H,W)
    clamped_tens = torch.clamp(denorm_tens, 0, 1)
    to_pil = transforms.ToPILImage() # tens<float>(3,H,W) -> img<int>(W,H,3)
    pil_img = to_pil(clamped_tens.cpu()) # img<int>(W,H,3)
    return pil_img


def calc_mean_std_loss(g,s):
    """
    tens<float>(N,C,H,W) tens<float>(N,C,H,W) -> tens<float>()
    given gen img feats and style img feats (for a particular chosen layer); returns style loss for this layer
    """
    g_mean = g.mean([2,3]) # (N,C)
    g_std = g.std([2,3]) # (N,C)
    s_mean = s.mean([2,3]) # (N,C)
    s_std = s.std([2,3]) # (N,C)
    style_loss = F.mse_loss(g_mean,s_mean) + F.mse_loss(g_std,s_std)
    return style_loss



if __name__ == "__main__":
    num_epochs = 2
    N = 8 # batch size
    lr = 1e-3 # learning rate
    wd = 1e-3 # weight decay
    lbda = 1 # style weight
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RandomPairDataset("AdaIN_Hayden/data/content/train", "AdaIN_Hayden/data/style/train")
    val_dataset = RandomPairDataset("AdaIN_Hayden/data/content/val", "AdaIN_Hayden/data/style/val")
    test_dataset = RandomPairDataset("AdaIN_Hayden/data/content/test", "AdaIN_Hayden/data/style/test")

    # for debugging / course tuning
    train_dataset = Subset(train_dataset, list(range(len(train_dataset)//10)))
    val_dataset = Subset(val_dataset, list(range(len(val_dataset)//10)))
    test_dataset = Subset(test_dataset, list(range(len(test_dataset)//10)))

    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=N, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=N, shuffle=False, num_workers=num_workers)

    enc = Encoder().to(device)
    adain = AdaIN().to(device)
    dec = Decoder().to(device)
    stn = StyleTransferNet(enc,adain,dec).to(device)
    opt = optim.AdamW(dec.parameters(), lr=lr, weight_decay=wd)

    num_exs_train = len(train_dataset)
    num_batches_train = math.ceil(num_exs_train/N)
    num_exs_val = len(val_dataset)
    num_batches_val = math.ceil(num_exs_val/N)
    num_exs_test = len(test_dataset)
    num_batches_test = math.ceil(num_exs_test/N)

    best_val_loss = float("inf")
    print("Entering epochs loop")
    for epoch in range(1,num_epochs+1):
        cumul_loss_c = 0
        cumul_loss_s = 0
        cumul_loss_tot = 0
        start_time = time.time()
        for batch_id,(c_img,s_img) in enumerate(train_loader,1): # (N,3,224,224) and (N,3,224,224)
            c_img = c_img.to(device)
            s_img = s_img.to(device)
            gen_img,t,s_11,s_21,s_31,s_41 = stn(c_img,s_img) # (N,3,224,224) (N,512,28,28) (N,64,224,224) (N,128,112,112) (N,256,56,56) (N,512,28,28)

            g_11,g_21,g_31,g_41 = enc(gen_img) # (N,64,224,224) (N,128,112,112) (N,256,56,56) (N,512,28,28)
            loss_c = F.mse_loss(g_41, t)
            loss_s = 0
            for g_feat,s_feat in zip([g_11,g_21,g_31,g_41], [s_11,s_21,s_31,s_41]): # relu1_1, relu2_1, relu3_1, relu4_1
                loss_s += calc_mean_std_loss(g_feat,s_feat)
            loss_tot = loss_c + lbda*loss_s

            opt.zero_grad()
            loss_tot.backward()
            opt.step()

            cumul_loss_c += loss_c.item()
            cumul_loss_s += loss_s.item()
            cumul_loss_tot += loss_tot.item()

            if batch_id%100==0:
                print(f"TRAIN: Epoch {epoch}/{num_epochs}, Batch {batch_id}/{num_batches_train}, Time {time.time()-start_time}, "
                      f"Avg L_c: {cumul_loss_c/batch_id:.4f}, Avg L_s: {cumul_loss_s/batch_id:.4f}, Avg L_tot: {cumul_loss_tot/batch_id:.4f}")
            # if batch_id==1000: break # NOTE: debugging only
        print(f"\nTRAIN: End of Epoch {epoch}/{num_epochs}, Avg L_c: {cumul_loss_c/batch_id:.4f}, Avg L_s: {cumul_loss_s/batch_id:.4f}, Avg L_tot: {cumul_loss_tot/batch_id:.4f}\n")


        cumul_loss_c = 0
        cumul_loss_s = 0
        cumul_loss_tot = 0
        start_time = time.time()
        with torch.no_grad():
            for batch_id,(c_img,s_img) in enumerate(val_loader,1): # (N,3,224,224) and (N,3,224,224)
                c_img = c_img.to(device)
                s_img = s_img.to(device)
                gen_img,t,s_11,s_21,s_31,s_41 = stn(c_img,s_img) # (N,3,224,224) (N,512,28,28) (N,64,224,224) (N,128,112,112) (N,256,56,56) (N,512,28,28)

                g_11,g_21,g_31,g_41 = enc(gen_img) # (N,64,224,224) (N,128,112,112) (N,256,56,56) (N,512,28,28)
                loss_c = F.mse_loss(g_41, t)
                loss_s = 0
                for g_feat,s_feat in zip([g_11,g_21,g_31,g_41], [s_11,s_21,s_31,s_41]): # relu1_1, relu2_1, relu3_1, relu4_1
                    loss_s += calc_mean_std_loss(g_feat,s_feat)
                loss_tot = loss_c + lbda*loss_s

                cumul_loss_c += loss_c.item()
                cumul_loss_s += loss_s.item()
                cumul_loss_tot += loss_tot.item()

                if batch_id%100==0:
                    print(f"VAL: Epoch {epoch}/{num_epochs}, Batch {batch_id}/{num_batches_val}, Time {time.time()-start_time}, "
                          f"Avg L_c: {cumul_loss_c/batch_id:.4f}, Avg L_s: {cumul_loss_s/batch_id:.4f}, Avg L_tot: {cumul_loss_tot/batch_id:.4f}")
        print(f"\nVAL: End of Epoch {epoch}/{num_epochs}, Avg L_c: {cumul_loss_c/batch_id:.4f}, Avg L_s: {cumul_loss_s/batch_id:.4f}, Avg L_tot: {cumul_loss_tot/batch_id:.4f}\n")
        avg_val_loss = cumul_loss_tot / batch_id
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_filename = f"decoder_epoch{epoch}_N{N}_lr{lr}_wd{wd}_lbda{lbda}_vloss{avg_val_loss:.4f}.pt"
            save_path = os.path.join("AdaIN_Hayden/checkpoints", model_filename)
            torch.save(dec.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
    

    # cumul_loss_c = 0
    # cumul_loss_s = 0
    # cumul_loss_tot = 0
    # start_time = time.time()
    # with torch.no_grad():
    #     for batch_id,(c_img,s_img) in enumerate(test_loader,1): # (N,3,224,224) and (N,3,224,224)
    #         c_img = c_img.to(device)
    #         s_img = s_img.to(device)
    #         gen_img,t,s_11,s_21,s_31,s_41 = stn(c_img,s_img) # (N,3,224,224) (N,512,28,28) (N,64,224,224) (N,128,112,112) (N,256,56,56) (N,512,28,28)

    #         g_11,g_21,g_31,g_41 = enc(gen_img) # (N,64,224,224) (N,128,112,112) (N,256,56,56) (N,512,28,28)
    #         loss_c = F.mse_loss(g_41, t)
    #         loss_s = 0
    #         for g_feat,s_feat in zip([g_11,g_21,g_31,g_41], [s_11,s_21,s_31,s_41]): # relu1_1, relu2_1, relu3_1, relu4_1
    #             loss_s += calc_mean_std_loss(g_feat,s_feat)
    #         loss_tot = loss_c + lbda*loss_s

    #         cumul_loss_c += loss_c.item()
    #         cumul_loss_s += loss_s.item()
    #         cumul_loss_tot += loss_tot.item()

    #         if batch_id%100==0:
    #             print(f"TEST: Epoch {epoch}/{num_epochs}, Batch {batch_id}/{num_batches_test}, Time {time.time()-start_time}, "
    #                     f"Avg L_c: {cumul_loss_c/batch_id:.4f}, Avg L_s: {cumul_loss_s/batch_id:.4f}, Avg L_tot: {cumul_loss_tot/batch_id:.4f}") 
    # print(f"\nTEST: End of final test, Avg L_c: {cumul_loss_c/batch_id:.4f}, Avg L_s: {cumul_loss_s/batch_id:.4f}, Avg L_tot: {cumul_loss_tot/batch_id:.4f}\n")






