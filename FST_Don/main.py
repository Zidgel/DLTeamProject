from fst_api import fst_api
from PIL import Image

if __name__ == "__main__":
    for i in range(5, 0, -1):
        sp = "../../fig1/style/s" + str(i) + ".jpg"
        style = Image.open(sp).convert("RGB")
        style = style.resize((180, 180))
        dataset = "../../content_images"
        api = fst_api(style, dataset, 5, 1e-4, 2e5)
        api.train_model()
        cp = "../../fig1/content/c" + str(i) + ".jpg"
        content = Image.open(cp).convert("RGB")
        path = "../../fig1/gen/cs" + str(i) + ".jpg"
        api.style_transfer(content, path)

    styles = ["s1.jpg", "s2.jpg", "s3.jpg", "s4.JPG", "s5.jpg", "s6.jpg"]
    content = ["c1.jpg", "c2.jpg", "c3.jpg"]
    s_pre = "../../fig2/style/"
    c_pre = "../../fig2/content/"

    for s in styles:
        for c in content:
            print("s: ", s_pre + s)
            style = Image.open(s_pre + s).convert("RGB")
            style = style.resize((180, 180))
            dataset = "../../content_images"
            api = fst_api(style, dataset, 5, 1e-4, 2e5)
            api.train_model()
            print("c: ", c_pre + c)
            content_image = Image.open(c_pre + c).convert("RGB")
            path = "../../fig2/gen/" + s[0:2] + c[0:2] + ".jpg"
            print("g", path)
            api.style_transfer(content_image, path)
