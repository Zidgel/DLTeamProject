# Hayden Schennum
# 2025-04-23

from AdaIN_train import *
from itertools import product



if __name__ == "__main__":
    model_to_load = "decoder_epoch2_N8_lr0.001_wd0.001_lbda1_vloss12.2429.pt"
    model_filepath = os.path.join("AdaIN_Hayden/checkpoints", model_to_load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    adain = AdaIN().to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(model_filepath))
    stn = StyleTransferNet(encoder, adain, decoder, alpha=1.0).to(device)
    stn.eval()

    content_dir = "AdaIN_Hayden/inference/content"
    style_dir = "AdaIN_Hayden/inference/style"
    content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]