import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
from itertools import cycle
from pathlib import Path
"""
NOT TOTALLY SURE ABOUT THE TRAINING PROCESS HERE REQUIRES ANALYSIS

"""
# ─────────────────── 1. Hyper‑params ───────────────────
BATCH_SIZE      = 1
#HAD TO DOWN SIZE THE IMAGE SIZE TO 124 from 256, too large for my GPU
IMAGE_SIZE      = 128
STYLE_TOKEN_DIM = 256
NUM_STYLE_TOKENS= 4
CONTENT_LAYERS  = 3
BASE_CHANNELS   = 64
NUM_BLOCKS      = 3
EPOCHS          = 10
LR              = 2e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────── 2. Data ───────────────────────────
tfm = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()])

# ▶  STYLE  — flat folder  ──────────────────────────────
class FlatImageFolder(Dataset):
    """Loads every *.jpg/ *.png inside a single folder as an image, ignores labels."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, root: str, transform=None):
        self.paths     = [p for p in Path(root).iterdir() if p.suffix.lower() in self.exts]
        assert self.paths, f"No images found in {root}"
        self.transform = transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img        # return image only, no label

style_ds = FlatImageFolder("../StyleData", tfm)

# CONTENT  (imagenet‑mini structure) ────────────────────
content_train = datasets.ImageFolder(root="../ContentData/train", transform=tfm)
content_val   = datasets.ImageFolder(root="../ContentData/val",   transform=tfm)
content_ds    = ConcatDataset([content_train, content_val])

# DataLoaders
style_dl   = DataLoader(style_ds,   batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
content_dl = DataLoader(content_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# helper to loop endlessly
def endless(dl):
    while True:
        for batch in dl:
            yield batch if isinstance(batch, torch.Tensor) else batch[0]

style_it, content_it = endless(style_dl), endless(content_dl)

# ─────────────────── 3. Model ──────────────────────────
from StyleShot import StyleEncoder, ContentExtraction, StyleTransfer

style_enc   = StyleEncoder(3, BASE_CHANNELS, STYLE_TOKEN_DIM, NUM_STYLE_TOKENS).to(DEVICE)
content_enc = ContentExtraction(3, CONTENT_LAYERS, fuse=True, downsample_rate=2).to(DEVICE)
decoder     = StyleTransfer(BASE_CHANNELS, STYLE_TOKEN_DIM, STYLE_TOKEN_DIM, NUM_BLOCKS).to(DEVICE)

optim  = torch.optim.Adam(
            list(style_enc.parameters()) +
            list(content_enc.parameters()) +
            list(decoder.parameters()), lr=LR)

l1 = nn.L1Loss()

# ─────────────────── 4. Training loop ──────────────────
for epoch in range(EPOCHS):
    for step in range(min(len(style_dl), len(content_dl))):
        style_img   = next(style_it).to(DEVICE)      # [B,3,256,256]
        content_img = next(content_it).to(DEVICE)

        #NOT SURE WHAT AUTOCAST DOES HERE
        with torch.cuda.amp.autocast():
            f_s = style_enc(style_img)               # [B,S,D]
            f_c = content_enc(content_img)           # [B,N,D]
            noise = torch.randn_like(content_img)    # same H×W
            out   = decoder(noise, f_s, f_c)

            loss_content = l1(out, content_img)
            loss_style   = l1(f_s.mean(1), f_c.mean(1))
            loss = loss_content + 0.1 * loss_style

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 25 == 0:
            print(f"[E{epoch+1}/{EPOCHS}  I{step}] "
                  f"L_total {loss.item():.3f}  Lc {loss_content.item():.3f}  Ls {loss_style.item():.3f}")

print("✅ Training finished")
