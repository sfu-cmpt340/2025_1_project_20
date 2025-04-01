import os
from PIL import Image, ImageEnhance, ImageOps
import random
from pathlib import Path

# --------- CONFIG ---------
INPUT_DIR = "../dermDatabaseOfficial/release_v0/images"
OUTPUT_DIR = "../augmented_dataset"
MIN_IMAGES_PER_CLASS = 30
AUGMENTATIONS = ["flip", "rotate", "brighten", "zoom"]
# --------------------------

def apply_augmentations(image):
    aug_images = []

    if "flip" in AUGMENTATIONS:
        aug_images.append(ImageOps.mirror(image))
    if "rotate" in AUGMENTATIONS:
        aug_images.append(image.rotate(90))
    if "brighten" in AUGMENTATIONS:
        enhancer = ImageEnhance.Brightness(image)
        aug_images.append(enhancer.enhance(1.5))
    if "zoom" in AUGMENTATIONS:
        w, h = image.size
        crop = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
        aug_images.append(crop.resize((w, h)))

    return aug_images

def augment_class(class_path, output_class_dir):
    images = list(Path(class_path).glob("*.jpg")) + list(Path(class_path).glob("*.png"))

    if len(images) >= MIN_IMAGES_PER_CLASS:
        return  # Skip already well-represented classes

    os.makedirs(output_class_dir, exist_ok=True)
    needed = MIN_IMAGES_PER_CLASS - len(images)
    i = 0
    while i < needed:
        img_path = random.choice(images)
        with Image.open(img_path) as img:
            aug_imgs = apply_augmentations(img)
            for aug in aug_imgs:
                if i >= needed:
                    break
                aug_filename = f"aug_{i}_{os.path.basename(img_path)}"
                aug.save(os.path.join(output_class_dir, aug_filename))
                i += 1

if __name__ == "__main__":
    class_folders = [d for d in Path(INPUT_DIR).iterdir() if d.is_dir()]

    for class_dir in class_folders:
        output_class_dir = Path(OUTPUT_DIR) / class_dir.name
        augment_class(class_dir, output_class_dir)

    print(f"✅ Augmentation complete. Output saved to: {OUTPUT_DIR}")