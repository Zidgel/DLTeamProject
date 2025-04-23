#!/usr/bin/env python3
"""
Neural-Style-Transfer (VGG-19, PyTorch) — cleaned-up reference
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# ────────────────────────── 0. Hyper-params ──────────────────────────
MAX_SIZE            = 680         # longest edge (pixels)
STYLE_WEIGHT        = 1e10
CONTENT_WEIGHT      = 1.0
TV_WEIGHT           = 1e-6
LR                  = 3e-3
LR_DECAY_EVERY      = 400         # iterations
LR_GAMMA            = 0.6
ITERATIONS          = 2000
CLAMP_RANGE         = (-2.5, 2.5) # roughly ±5 σ in ImageNet space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────── 1. Pre-/post-processing ──────────────────────
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

preprocess = transforms.Compose([
    transforms.Resize(MAX_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(MAX_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean.squeeze(), std=imagenet_std.squeeze()),
])

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

def denormalize(img: torch.Tensor) -> torch.Tensor:
    return (img.cpu() * imagenet_std + imagenet_mean).clamp(0, 1)

def show(img: torch.Tensor, title: str = "") -> None:
    plt.imshow(transforms.ToPILImage()(denormalize(img.squeeze(0))))
    plt.title(title)
    plt.axis("off")
    plt.show()

# ───────────────────────── 2. VGG encoder ────────────────────────────
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
for idx, m in enumerate(vgg):
    if isinstance(m, torch.nn.MaxPool2d):
        vgg[idx] = torch.nn.AvgPool2d(kernel_size=m.kernel_size,
                                      stride=m.stride,
                                      padding=m.padding)
vgg.eval().requires_grad_(False).to(device)

layer_map = {
    '0':  'conv1_1',
    '5':  'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'content',      # conv4_2 in the original Gatys et al.
    '28': 'conv5_1',
}

def extract_feats(x: torch.Tensor):
    feats = {}
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layer_map:
            feats[layer_map[name]] = x
    return feats

def gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    x = x.reshape(c, h * w)
    return x @ x.t() / (c * h * w)

# ─────────────────────────── 3. Losses ───────────────────────────────
def style_loss(tgt, sty):
    return sum(F.mse_loss(gram(tgt[l]), gram(sty[l])) for l in sty)

def content_loss(tgt, cont):
    return F.mse_loss(tgt['content'], cont['content'])

def tv_loss(img):
    x_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
    y_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
    return F.l1_loss(x_diff, torch.zeros_like(x_diff)) + \
           F.l1_loss(y_diff, torch.zeros_like(y_diff))

# ─────────────────────────── 4. Main loop ────────────────────────────
def neural_style_transfer(content_path: str, style_path: str,
                          out_path: str = "stylised.png") -> None:

    content = load_image(content_path)
    style   = load_image(style_path)

    with torch.no_grad():
        content_feats = extract_feats(content)
        style_feats   = extract_feats(style)

    target = content.clone().requires_grad_(True)

    optimiser = optim.Adam([target], lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                          step_size=LR_DECAY_EVERY,
                                          gamma=LR_GAMMA)

    for it in range(ITERATIONS + 1):
        optimiser.zero_grad()

        t_feats = extract_feats(target)
        s_loss  = style_loss(t_feats, style_feats)
        c_loss  = content_loss(t_feats, content_feats)
        tvl     = tv_loss(target)

        loss = (STYLE_WEIGHT * s_loss +
                CONTENT_WEIGHT * c_loss +
                TV_WEIGHT * tvl)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            target.clamp_(*CLAMP_RANGE)
        scheduler.step()

        if it % 100 == 0 or it == ITERATIONS:
            print(f"[{it:4d}/{ITERATIONS}] "
                  f"style={s_loss:8.4f}  content={c_loss:8.4f}  tv={tvl:8.4f}")

    # ──────────────────── 5. Save / display result ──────────────────
    img_out = denormalize(target.squeeze(0))
    out_pil = transforms.ToPILImage()(img_out)
    out_pil.save(out_path)
    print(f"✓ Saved → {Path(out_path).resolve()}")
    show(target, "Stylised Result")

# ──────────────────────────── 6. Entry ──────────────────────────────
if __name__ == "__main__":
    neural_style_transfer(
        content_path="../ContentImages/ILSVRC2012_test_00000004.JPEG",
        style_path="../StyleData/panel_99.png",
        out_path="stylised_output.png",
    )
