import matplotlib.pyplot as plt
from PIL import Image


def resize_height(img, target_width=784, target_height=512):
    w, h = img.size
    new_width = int(w * (target_height / h))
    img = img.resize((new_width, target_height))
    print(new_width, target_height)

    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    offset = ((target_width - new_width) // 2, 0)
    background.paste(img, offset)

    return background


def plot_image_grid(images, rows, cols, path, col_titles=None, figsize=(24, 14)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    i = 0
    for col in range(cols):
        for row in range(rows):
            ax = axes[row, col]

            if i < len(images):
                ax.imshow(images[i])
            ax.axis("off")
            i += 1

    if col_titles is not None:
        for col in range(cols):
            axes[0, col].set_title(col_titles[col], fontsize=42, pad=15)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(path)


if __name__ == "__main__":
    paths = [
        "FST_Don/fig1/content/c1.jpg",
        "FST_Don/fig1/content/c2.jpg",
        "FST_Don/fig1/content/c3.jpg",
        "FST_Don/fig1/content/c4.jpg",
        "FST_Don/fig1/content/c5.jpg",
        "FST_Don/fig1/style/s1.jpg",
        "FST_Don/fig1/style/s2.jpg",
        "FST_Don/fig1/style/s3.jpg",
        "FST_Don/fig1/style/s4.jpg",
        "FST_Don/fig1/style/s5.jpg",
        "data/fig1_generated/NST/g1n.png",
        "data/fig1_generated/NST/g2n.png",
        "data/fig1_generated/NST/g3n.png",
        "data/fig1_generated/NST/g4n.png",
        "data/fig1_generated/NST/g5n.png",
        "data/fig1_generated/FST/g1f.jpg",
        "data/fig1_generated/FST/g2f.jpg",
        "data/fig1_generated/FST/g3f.jpg",
        "data/fig1_generated/FST/g4f.jpg",
        "data/fig1_generated/FST/g5f.jpg",
        "data/fig1_generated/AdaIN/g1a.png",
        "data/fig1_generated/AdaIN/g2a.png",
        "data/fig1_generated/AdaIN/g3a.png",
        "data/fig1_generated/AdaIN/g4a.png",
        "data/fig1_generated/AdaIN/g5a.png",
        "data/fig1_generated/StyleShot/g1s.jpg",
        "data/fig1_generated/StyleShot/g2s.jpg",
        "data/fig1_generated/StyleShot/g3s.jpg",
        "data/fig1_generated/StyleShot/g4s.jpg",
        "data/fig1_generated/StyleShot/g5s.jpg",
    ]

    plot_image_grid(
        [resize_height(Image.open(img)) for img in paths],
        5,
        6,
        "data/fig1_generated/figure1_1",
        col_titles=["Content", "Style", "NST", "FST", "AdaIN", "StyleShot"],
    )

    # Content 1
    paths = [
        "FST_Don/fig2/content/c1.jpg",
        "FST_Don/fig2/content/c1.jpg",
        "FST_Don/fig2/content/c1.jpg",
        "FST_Don/fig2/content/c1.jpg",
        "FST_Don/fig2/content/c1.jpg",
        "FST_Don/fig2/content/c1.jpg",
        "FST_Don/fig2/style/s1.jpg",
        "FST_Don/fig2/style/s2.jpg",
        "FST_Don/fig2/style/s3.jpg",
        "FST_Don/fig2/style/s4.jpg",
        "FST_Don/fig2/style/s5.jpg",
        "FST_Don/fig2/style/s6.jpg",
        "data/fig2_generated/NST/s1.jpg_c.jpg.png",
        "data/fig2_generated/NST/s2.jpg_c.jpg.png",
        "data/fig2_generated/NST/s3.jpg_c.jpg.png",
        "data/fig2_generated/NST/s4.jpg_c.jpg.png",
        "data/fig2_generated/NST/s5.jpg_c.jpg.png",
        "data/fig2_generated/NST/s6.jpg_c.jpg.png",
        "data/fig2_generated/FST/s1c1.jpg",
        "data/fig2_generated/FST/s2c1.jpg",
        "data/fig2_generated/FST/s3c1.jpg",
        "data/fig2_generated/FST/s4c1.jpg",
        "data/fig2_generated/FST/s5c1.jpg",
        "data/fig2_generated/FST/s6c1.jpg",
        "data/fig2_generated/AdaIN/g1a.png",
        "data/fig2_generated/AdaIN/g2a.png",
        "data/fig2_generated/AdaIN/g3a.png",
        "data/fig2_generated/AdaIN/g4a.png",
        "data/fig2_generated/AdaIN/g5a.png",
        "data/fig2_generated/AdaIN/g6a.png",
        "data/fig2_generated/StyleShot/s1.jpg_c1.jpg",
        "data/fig2_generated/StyleShot/s2.jpg_c1.jpg",
        "data/fig2_generated/StyleShot/s3.jpg_c1.jpg",
        "data/fig2_generated/StyleShot/s4.jpg_c1.jpg",
        "data/fig2_generated/StyleShot/s5.jpg_c1.jpg",
        "data/fig2_generated/StyleShot/s6.jpg_c1.jpg",
    ]

    plot_image_grid(
        [resize_height(Image.open(img)) for img in paths],
        6,
        6,
        "data/fig2_generated/figure2_1",
        col_titles=["Content", "Style", "NST", "FST", "AdaIN", "StyleShot"],
    )

    # Content 2
    paths = [
        "FST_Don/fig2/content/c2.jpg",
        "FST_Don/fig2/content/c2.jpg",
        "FST_Don/fig2/content/c2.jpg",
        "FST_Don/fig2/content/c2.jpg",
        "FST_Don/fig2/content/c2.jpg",
        "FST_Don/fig2/content/c2.jpg",
        "FST_Don/fig2/style/s1.jpg",
        "FST_Don/fig2/style/s2.jpg",
        "FST_Don/fig2/style/s3.jpg",
        "FST_Don/fig2/style/s4.jpg",
        "FST_Don/fig2/style/s5.jpg",
        "FST_Don/fig2/style/s6.jpg",
        "data/fig2_generated/NST/s1.jpg_c2.jpg.png",
        "data/fig2_generated/NST/s2.jpg_c2.jpg.png",
        "data/fig2_generated/NST/s3.jpg_c2.jpg.png",
        "data/fig2_generated/NST/s4.jpg_c2.jpg.png",
        "data/fig2_generated/NST/s5.jpg_c2.jpg.png",
        "data/fig2_generated/NST/s6.jpg_c2.jpg.png",
        "data/fig2_generated/FST/s1c2.jpg",
        "data/fig2_generated/FST/s2c2.jpg",
        "data/fig2_generated/FST/s3c2.jpg",
        "data/fig2_generated/FST/s4c2.jpg",
        "data/fig2_generated/FST/s5c2.jpg",
        "data/fig2_generated/FST/s6c2.jpg",
        "data/fig2_generated/AdaIN/g7a.png",
        "data/fig2_generated/AdaIN/g8a.png",
        "data/fig2_generated/AdaIN/g9a.png",
        "data/fig2_generated/AdaIN/g10a.png",
        "data/fig2_generated/AdaIN/g11a.png",
        "data/fig2_generated/AdaIN/g12a.png",
        "data/fig2_generated/StyleShot/s1.jpg_c2.jpg",
        "data/fig2_generated/StyleShot/s2.jpg_c2.jpg",
        "data/fig2_generated/StyleShot/s3.jpg_c2.jpg",
        "data/fig2_generated/StyleShot/s4.jpg_c2.jpg",
        "data/fig2_generated/StyleShot/s5.jpg_c2.jpg",
        "data/fig2_generated/StyleShot/s6.jpg_c2.jpg",
    ]

    plot_image_grid(
        [resize_height(Image.open(img)) for img in paths],
        6,
        6,
        "data/fig2_generated/figure2_2",
        col_titles=["Content", "Style", "NST", "FST", "AdaIN", "StyleShot"],
    )

    # Content 3
    paths = [
        "FST_Don/fig2/content/c3.jpg",
        "FST_Don/fig2/content/c3.jpg",
        "FST_Don/fig2/content/c3.jpg",
        "FST_Don/fig2/content/c3.jpg",
        "FST_Don/fig2/content/c3.jpg",
        "FST_Don/fig2/content/c3.jpg",
        "FST_Don/fig2/style/s1.jpg",
        "FST_Don/fig2/style/s2.jpg",
        "FST_Don/fig2/style/s3.jpg",
        "FST_Don/fig2/style/s4.jpg",
        "FST_Don/fig2/style/s5.jpg",
        "FST_Don/fig2/style/s6.jpg",
        "data/fig2_generated/NST/s1.jpg_c3.jpg.png",
        "data/fig2_generated/NST/s2.jpg_c3.jpg.png",
        "data/fig2_generated/NST/s3.jpg_c3.jpg.png",
        "data/fig2_generated/NST/s4.jpg_c3.jpg.png",
        "data/fig2_generated/NST/s5.jpg_c3.jpg.png",
        "data/fig2_generated/NST/s6.jpg_c3.jpg.png",
        "data/fig2_generated/FST/s1c3.jpg",
        "data/fig2_generated/FST/s2c3.jpg",
        "data/fig2_generated/FST/s3c3.jpg",
        "data/fig2_generated/FST/s4c3.jpg",
        "data/fig2_generated/FST/s5c3.jpg",
        "data/fig2_generated/FST/s6c3.jpg",
        "data/fig2_generated/AdaIN/g13a.png",
        "data/fig2_generated/AdaIN/g14a.png",
        "data/fig2_generated/AdaIN/g15a.png",
        "data/fig2_generated/AdaIN/g16a.png",
        "data/fig2_generated/AdaIN/g17a.png",
        "data/fig2_generated/AdaIN/g18a.png",
        "data/fig2_generated/StyleShot/s1.jpg_c3.jpg",
        "data/fig2_generated/StyleShot/s2.jpg_c3.jpg",
        "data/fig2_generated/StyleShot/s3.jpg_c3.jpg",
        "data/fig2_generated/StyleShot/s4.jpg_c3.jpg",
        "data/fig2_generated/StyleShot/s5.jpg_c3.jpg",
        "data/fig2_generated/StyleShot/s6.jpg_c3.jpg",
    ]

    plot_image_grid(
        [resize_height(Image.open(img)) for img in paths],
        6,
        6,
        "data/fig2_generated/figure2_3",
        col_titles=["Content", "Style", "NST", "FST", "AdaIN", "StyleShot"],
    )
