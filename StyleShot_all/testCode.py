import torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
from itertools import cycle



HED_dataset = datasets.BSDS500(root="../HED-BSDS500", split="train", download=True)
"""
NOT TOTALLY SURE ABOUT THE TRAINING PROCESS HERE REQUIRES ANALYSIS

"""
def denorm(x):                 # x is B×3×H×W in [-1,1]
    return x * 0.5 + 0.5
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
LR           = 2e-4
CKPT_DIR     = Path("./checkpoints")        
CKPT_DIR.mkdir(exist_ok=True)
DEBUG_DIR    = Path("./debug")              
DEBUG_DIR.mkdir(exist_ok=True)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────── 2. Datasets & loaders ───────

#LOOK INTO THIS TRANSFORM:
tfm = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

class FlatImageFolder(Dataset):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, root, transform=None):
        self.paths     = [p for p in Path(root).iterdir() if p.suffix.lower() in self.exts]
        self.transform = transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0   # return dummy label

style_dl = DataLoader(FlatImageFolder("../StyleData", tfm),
                      batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

content_train = datasets.ImageFolder("../ContentData/train", tfm)
content_val   = datasets.ImageFolder("../ContentData/val",   tfm)
content_ds    = ConcatDataset([content_train, content_val])
content_dl = DataLoader(content_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def endless(dl):
    while True:
        for x, *_ in dl:         # ImageFolder returns (img,label)
            yield x

style_it, content_it = endless(style_dl), endless(content_dl)

# ───────────── 3. Model ─────────────────────
from StyleShot import StyleEncoder, ContentExtraction, StyleTransfer
style_enc   = StyleEncoder(3, 64, 256, 4).to(DEVICE)
content_enc = ContentExtraction(3, 3, fuse=True, downsample_rate=2).to(DEVICE)
decoder     = StyleTransfer(64, 256, 256, 3).to(DEVICE)

optim = torch.optim.Adam(
            list(style_enc.parameters()) +
            list(content_enc.parameters()) +
            list(decoder.parameters()), lr=LR)

l1 = nn.L1Loss()

# ───────────── 4. Training + save / preview ─
for epoch in range(EPOCHS):
    style_enc.train(); content_enc.train(); decoder.train()

    for step in range(min(len(style_dl), len(content_dl))):
        style_img   = next(style_it).to(DEVICE)
        content_img = next(content_it).to(DEVICE)
        noise       = torch.randn_like(content_img)

        with torch.cuda.amp.autocast():
            f_s   = style_enc(style_img)
            f_c   = content_enc(content_img)
            out   = decoder(noise, f_s, f_c)

            loss_content = l1(out, content_img)
            loss_style   = l1(f_s.mean(1), f_c.mean(1))
            loss         = loss_content + 0.1 * loss_style

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 25 == 0:
            print(f"[E{epoch+1}/{EPOCHS}  I{step}] "
                  f"L_total {loss.item():.3f}  Lc {loss_content.item():.3f}  Ls {loss_style.item():.3f}")

    # ─── END‑OF‑EPOCH: checkpoint + preview ───
    ckpt_path = CKPT_DIR / f"styleshot_epoch{epoch+1}.pth"
    torch.save({
        "epoch": epoch+1,
        "style_enc":   style_enc.state_dict(),
        "content_enc": content_enc.state_dict(),
        "decoder":     decoder.state_dict(),
        "optim":       optim.state_dict()
    }, ckpt_path)
    print(f"✅  Saved checkpoint to {ckpt_path}")

    # ---------- SANITY CHECK  (preview image) ----------
    style_enc.eval(); content_enc.eval(); decoder.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        # grab *one* mini‑batch for the preview
        style_img   = next(style_it).to(DEVICE)
        content_img = next(content_it).to(DEVICE)
        noise       = torch.randn_like(content_img)
        preview_out = decoder(noise,
                              style_enc(style_img),
                              content_enc(content_img))

    torchvision.utils.save_image(
        denorm(preview_out).clamp(0,1),          # use raw preview_out if you applied Fix B
        DEBUG_DIR / f"epoch{epoch+1}_preview.jpg",
        nrow=4)
    print(f"🖼️  Wrote preview to debug/epoch{epoch+1}_preview.jpg")

print("🎉 Training complete")
