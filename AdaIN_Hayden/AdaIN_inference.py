# Hayden Schennum
# 2025-04-23
# note to self: next time, use pil<int>(shape) instead of img<int>(shape)

from AdaIN_train import *
from itertools import product
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg_mean, std=vgg_std),
        ]) # img<int>(W,H,3) -> tens<float>(3,H,W)



if __name__ == "__main__":
    alpha = 1.0
    model_to_load = "decoder_epoch2_N4_lr0.001_wd0.001_lbda0.1_Lc6.5507_Ls4.3967_Ltot6.9903.pt"

    model_filepath = os.path.join("AdaIN_Hayden/checkpoints", model_to_load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    adain = AdaIN().to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(model_filepath))
    stn = StyleTransferNet(encoder, adain, decoder, alpha=alpha).to(device)
    stn.eval()

    content_dir = "AdaIN_Hayden/inference/content"
    style_dir = "AdaIN_Hayden/inference/style"
    grid_file = "AdaIN_Hayden/inference/generated_grid.png"
    content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]
    content_img_list = [Image.open(p).convert("RGB") for p in content_paths] # list<img<int>(W,H,3)>
    style_img_list = [Image.open(p).convert("RGB") for p in style_paths] # list<img<int>(W,H,3)>
    content_tens_list = [norm_transform(img) for img in content_img_list] # list<tens<float>(3,H,W)>>
    style_tens_list = [norm_transform(img) for img in style_img_list] # list<tens<float>(3,H,W)>>

    print("Entering image gen loop")
    gen_img_list = [] # list<img<int>(W,H,3)>
    for c,s in product(content_tens_list, style_tens_list): # (3,H,W) and (3,H,W)
        c_4d = c.unsqueeze(0).to(device) # (1,3,H,W)
        s_4d = s.unsqueeze(0).to(device) # (1,3,H,W)
        with torch.no_grad():
            gen_4d, *_ = stn(c_4d,s_4d) # (1,3,H,W)
        gen = gen_4d.squeeze(0) # (3,H,W)
        gen_img = norm_tens_to_denorm_img(gen) # img<int>(W,H,3)
        gen_img_list.append(gen_img)
    print("Leaving image gen loop")
    
    fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(12, 12))
    for row in range(0,6):
        for col in range(0,6):
            ax = axs[row,col]
            if row==0 and col==0:
                ax.axis('off')
            elif row==0: # style images in first row
                style_img = np.array(style_img_list[col-1]) # arr<int>(H,W,3)
                ax.imshow(style_img)
                ax.set_title(f"Style {col}", fontsize=8)
                ax.axis('off')
            elif col==0: # content images in first col
                content_img = np.array(content_img_list[row-1]) # arr<int>(H,W,3)
                ax.imshow(content_img)
                ax.set_title(f"Content {row}", fontsize=8)
                ax.axis('off')
            else: # generated images in inner cells
                idx = 5*(row-1) + (col-1)
                gen_img = np.array(gen_img_list[idx]) # arr<int>(H,W,3)
                ax.imshow(gen_img)
                ax.axis('off')    
    plt.tight_layout()
    plt.savefig(grid_file)
    print("Grid image saved")
