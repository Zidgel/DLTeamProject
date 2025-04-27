from fst_api import fst_api
from PIL import Image
import torch

if __name__ == "__main__":
    for i in range(5, 0, -1):
        sp = "fig1/style/s" + str(i) + ".jpg"
        style = Image.open(sp).convert("RGB")
        # style = style.resize((600, 600))
        dataset = "../../imagenette2-320"
        api = fst_api(style, dataset, 4, 1e-3, 1e5)
        api.train_model()
        cp = "fig1/content/c" + str(i) + ".jpg"
        content = Image.open(cp).convert("RGB")
        path = "fig1/gen/cs" + str(i) + ".jpg"
        api.style_transfer(content, path)
        del api
        torch.cuda.empty_cache()

    styles = ["s1.jpg", "s2.jpg", "s3.jpg", "s4.JPG", "s5.jpg", "s6.jpg"]
    content = ["c1.jpg", "c2.jpg", "c3.jpg"]
    s_pre = "fig2/style/"
    c_pre = "fig2/content/"

    for s in styles:
        print("s: ", s_pre + s)
        style = Image.open(s_pre + s).convert("RGB")
        # style = style.resize((1000, 1000))
        dataset = "../../imagenette2-320"
        api = fst_api(style, dataset, 4, 1e-3, 1e5)
        api.train_model()
        for c in content:
            print("c: ", c_pre + c)
            content_image = Image.open(c_pre + c).convert("RGB")
            path = "fig2/gen/" + s[0:2] + c[0:2] + ".jpg"
            print("g", path)
            api.style_transfer(content_image, path)

        del api
        torch.cuda.empty_cache()
