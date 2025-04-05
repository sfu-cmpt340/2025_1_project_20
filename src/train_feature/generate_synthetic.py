import os
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from pathlib import Path

# -------- CONFIG --------
INPUT_DIR = "../dermDatabaseOfficial/release_v0/images"
OUTPUT_DIR = "../synthetic_dataset"
TARGET_CLASSES = ["Fml", "Gcl", "Gdl", "Gbl", "NHL"]
SYNTHETIC_COUNT = 30  # per class
# ------------------------

# Ensure same results across different runs
# NOTE: this seed was not used in the project documentation
random.seed(100)

def generate_variants(image):
    variants = []

    # Flip
    variants.append(ImageOps.mirror(image))

    # Rotate randomly
    variants.append(image.rotate(random.choice([15, 30, 45])))

    # Color enhancement
    enhancer = ImageEnhance.Color(image)
    variants.append(enhancer.enhance(random.uniform(1.2, 1.5)))

    # Blur
    variants.append(image.filter(ImageFilter.GaussianBlur(radius=1)))

    # Zoom (crop and resize)
    w, h = image.size
    crop = image.crop((w*0.1, h*0.1, w*0.9, h*0.9))
    variants.append(crop.resize((w, h)))

    return variants

def generate_synthetic_for_class(class_name):
    class_path = Path(INPUT_DIR) / class_name
    print(f"🔍 Checking class: {class_name} at {class_path}")

    # Gather all image extensions
    images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
    print(f"📸 Found {len(images)} images in {class_name}")

    if not images:
        print(f"⚠️ No images found for class {class_name}")
        return

    output_path = Path(OUTPUT_DIR) / class_name
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    while count < SYNTHETIC_COUNT:
        img_path = random.choice(images)
        with Image.open(img_path) as img:
            variants = generate_variants(img)
            for var in variants:
                if count >= SYNTHETIC_COUNT:
                    break
                out_name = f"syn_{count}_{img_path.name}"
                var.save(output_path / out_name)
                count += 1

if __name__ == "__main__":
    print("📂 Available class folders in dataset:")
    for folder in Path(INPUT_DIR).iterdir():
        if folder.is_dir():
            print(" -", folder.name)

    for cls in TARGET_CLASSES:
        generate_synthetic_for_class(cls)

    print(f"✅ Synthetic images saved to: {OUTPUT_DIR}")