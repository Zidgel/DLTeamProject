from fst_api import fst_api
from PIL import Image

if __name__ == "__main__":
    style = Image.open("../../OCR_PARSED_PANELS/panel_99.png").convert("RGB")
    dataset = "../../content_images"
    api = fst_api(style, dataset, 5, 1e-4, 1e5)
    api.train_model()
    content = Image.open("../../content_images/bron.png").convert("RGB")
    api.style_transfer(content)
