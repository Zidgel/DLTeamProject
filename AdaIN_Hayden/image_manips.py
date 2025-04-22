
import os
from PIL import Image
from shutil import copy2

def find_smallest_dimension(folder_path):
    smallest = float('inf')
    smallest_name = ""
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            with Image.open(os.path.join(root, file)) as img:
                width, height = img.size
                if width<smallest or height<smallest:
                    smallest = min(width,height)
                    smallest_name = file
                count += 1
            # if count%1000 == 0:
            #     print(count)
    print(f"Smallest dimension among all images: {smallest}")
    return smallest


def filter_and_remove_small_images(dataset_dir, min_dim, move_to_dir):
    removed_count = 0
    for filename in os.listdir(dataset_dir):
        filepath = os.path.join(dataset_dir, filename)
        try:
            with Image.open(filepath) as img:
                w, h = img.size
            if min(w, h) < min_dim:
                dest_path = os.path.join(move_to_dir, filename)
                copy2(filepath, dest_path)  # Copy first
                os.remove(filepath)       # Then delete original
                removed_count += 1
                
        except Exception as e:
            print(f"Skipping corrupted image {filename}: {e}")

    print(f"Removed/moved {removed_count} images with dimensions < {min_dim}px.")


if __name__ == "__main__":

    # filter_and_remove_small_images("AdaIN_Hayden/data/content/train", 224, "AdaIN_Hayden/data/content/train_toosmall")
    # filter_and_remove_small_images("AdaIN_Hayden/data/content/val", 224, "AdaIN_Hayden/data/content/val_toosmall")
    # filter_and_remove_small_images("AdaIN_Hayden/data/content/test", 224, "AdaIN_Hayden/data/content/test_toosmall")

    # smallest_dim = find_smallest_dimension("AdaIN_Hayden/data/content/train")
    # smallest_dim = find_smallest_dimension("AdaIN_Hayden/data/content/val")
    # smallest_dim = find_smallest_dimension("AdaIN_Hayden/data/content/test")

    smallest_dim = find_smallest_dimension("AdaIN_Hayden/data/style/train")
    smallest_dim = find_smallest_dimension("AdaIN_Hayden/data/style/val")
    smallest_dim = find_smallest_dimension("AdaIN_Hayden/data/style/test")

