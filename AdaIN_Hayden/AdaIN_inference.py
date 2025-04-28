# Hayden Schennum
# 2025-04-23
# note to self: next time, use pil<int>(shape) instead of img<int>(shape)

from AdaIN_train import *
from itertools import product
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


S = 448 # side of square center crop
resize_transform = transforms.Compose([
            transforms.Lambda(partial(resize_if_larger, thresh=S)),
            transforms.CenterCrop(S),
        ]) # img<int>(W,H,3) -> img<int>(S,S,3)
norm_transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg_mean, std=vgg_std),
        ]) # img<int>(W,H,3) -> tens<float>(3,S,S)


nocrop_norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg_mean, std=vgg_std),
        ]) # img<int>(W,H,3) -> tens<float>(3,H,W)



if __name__ == "__main__":
    alpha = 1 # style focus (inference)
    model_to_load = "decoder_epoch8_N4_lr0.0001_wd0.0001_lbda10_Lc9.0121_Ls1.1444_Ltot20.4565.pt"

    model_filepath = os.path.join("AdaIN_Hayden/checkpoints", model_to_load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    adain = AdaIN().to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(model_filepath))
    stn = StyleTransferNet(encoder, adain, decoder, alpha=alpha).to(device)
    stn.eval()



    content_dir = "AdaIN_Hayden/inference/fig1/content"
    style_dir = "AdaIN_Hayden/inference/fig1/style"
    generated_dir = "AdaIN_Hayden/inference/fig1/generated"
    content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]
    content_img_list = [Image.open(p).convert("RGB") for p in content_paths] # list<img<int>(W,H,3)>
    style_img_list = [Image.open(p).convert("RGB") for p in style_paths] # list<img<int>(W,H,3)>
    content_tens_list = [nocrop_norm_transform(img) for img in content_img_list] # list<tens<float>(3,H,W)>>
    style_tens_list = [nocrop_norm_transform(img) for img in style_img_list] # list<tens<float>(3,H,W)>>

    content_style_pairs = list(zip(content_tens_list, style_tens_list)) # list<(tens<float>(3,H,W)>,tens<float>(3,H,W)>)>
    with torch.no_grad():
        for i,(c_3d,s_3d) in enumerate(content_style_pairs):
            c_4d = c_3d.unsqueeze(0).to(device)
            s_4d = s_3d.unsqueeze(0).to(device)
            gen_4d, *_ = stn(c_4d,s_4d) # (1,3,H,W)
            gen_3d = gen_4d.squeeze(0) # (3,H,W)
            gen_img = norm_tens_to_denorm_img(gen_3d.cpu()) # img<int>(W,H,3)
            save_path = os.path.join(generated_dir, f"g{i+1}a.png")
            gen_img.save(save_path)
        


    # content_dir = "AdaIN_Hayden/inference/fig2/content"
    # style_dir = "AdaIN_Hayden/inference/fig2/style"
    # generated_dir = "AdaIN_Hayden/inference/fig2/generated"
    # content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    # style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]
    # content_img_list = [Image.open(p).convert("RGB") for p in content_paths] # list<img<int>(W,H,3)>
    # style_img_list = [Image.open(p).convert("RGB") for p in style_paths] # list<img<int>(W,H,3)>
    # content_tens_list = [nocrop_norm_transform(img) for img in content_img_list] # list<tens<float>(3,H,W)>>
    # style_tens_list = [nocrop_norm_transform(img) for img in style_img_list] # list<tens<float>(3,H,W)>>

    # content_style_pairs = list(product(content_tens_list, style_tens_list)) # list<(tens<float>(3,H,W)>,tens<float>(3,H,W)>)>
    # with torch.no_grad():
    #     for i,(c_3d,s_3d) in enumerate(content_style_pairs):
    #         c_4d = c_3d.unsqueeze(0).to(device)
    #         s_4d = s_3d.unsqueeze(0).to(device)
    #         gen_4d, *_ = stn(c_4d,s_4d) # (1,3,H,W)
    #         gen_3d = gen_4d.squeeze(0) # (3,H,W)
    #         gen_img = norm_tens_to_denorm_img(gen_3d.cpu()) # img<int>(W,H,3)
    #         save_path = os.path.join(generated_dir, f"g{i+1}a.png")
    #         gen_img.save(save_path)




    # content_dir = "AdaIN_Hayden/inference/grid/content"
    # style_dir = "AdaIN_Hayden/inference/grid/style"
    # grid_file = "AdaIN_Hayden/inference/grid/generated_grid.png"
    # content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    # style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]
    # content_img_list = [Image.open(p).convert("RGB") for p in content_paths] # list<img<int>(W,H,3)>
    # style_img_list = [Image.open(p).convert("RGB") for p in style_paths] # list<img<int>(W,H,3)>
    # content_tens_list = [norm_transform(img) for img in content_img_list] # list<tens<float>(3,H,W)>>
    # style_tens_list = [norm_transform(img) for img in style_img_list] # list<tens<float>(3,H,W)>>
    
    # content_style_pairs = list(product(content_tens_list, style_tens_list)) # list<(tens<float>(3,H,W)>,tens<float>(3,H,W)>)>
    # content_batch = torch.stack([pair[0] for pair in content_style_pairs]).to(device) # (N,3,H,W)
    # style_batch = torch.stack([pair[1] for pair in content_style_pairs]).to(device) # (N,3,H,W)
    # with torch.no_grad():
    #     gen_batch, *_ = stn(content_batch,style_batch) # (N,3,H,W)
    # gen_img_list = [] # list<img<int>(W,H,3)>
    # for i in range(0,gen_batch.size(0)): 
    #     gen_img = norm_tens_to_denorm_img(gen_batch[i].cpu()) # img<int>(W,H,3)
    #     gen_img_list.append(gen_img)
    
    # fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(12, 12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    # for row in range(0,6):
    #     for col in range(0,6):
    #         ax = axs[row,col]
    #         if row==0 and col==0:
    #             ax.axis('off')
    #         elif row==0: # content images in first row
    #             content_img = content_img_list[col-1] # img<int>(W,H,3)
    #             content_img = resize_transform(content_img) # img<int>(224,224,3)
    #             content_img = np.array(content_img) # arr<int>(224,224,3)
    #             ax.imshow(content_img)
    #             # ax.set_title(f"Content {col}", fontsize=8)
    #             ax.axis('off')
    #         elif col==0: # style images in first col
    #             style_img = style_img_list[row-1] # img<int>(W,H,3)
    #             style_img = resize_transform(style_img) # img<int>(224,224,3)
    #             style_img = np.array(style_img) # arr<int>(224,224,3)
    #             ax.imshow(style_img)
    #             # ax.set_title(f"Style {row}", fontsize=8)
    #             ax.axis('off')
    #         else: # generated images in inner cells
    #             # idx = 5*(row-1) + (col-1)
    #             idx = 5*(col-1) + (row-1)
    #             gen_img = np.array(gen_img_list[idx]) # arr<int>(H,W,3)
    #             ax.imshow(gen_img)
    #             ax.axis('off')
    # plt.savefig(grid_file, bbox_inches='tight', pad_inches=0)
    # print("Grid image saved")




