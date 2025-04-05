import os
import torch
from pytorch_fid import fid_score
import glob
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
#its about how to flat the images
def flatten_images(src_dir, dest_dir, resize=(299, 299)):
    os.makedirs(dest_dir, exist_ok=True)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in Path(src_dir).rglob(ext):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(resize)
                target_path = Path(dest_dir) / img_path.name
                img.save(target_path)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

def compute_fid(real_dir, fake_dir, dims=2048):
    temp_real = "__temp_eval/real"
    temp_fake = "__temp_eval/fake"
    shutil.rmtree("__temp_eval", ignore_errors=True)
    flatten_images(real_dir, temp_real)
    flatten_images(fake_dir, temp_fake)

    real_files = sorted(glob.glob(os.path.join(temp_real, "*.*")))
    fake_files = sorted(glob.glob(os.path.join(temp_fake, "*.*")))

    print("Number of real images:", len(real_files))
    print("Number of fake images:", len(fake_files))

    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError("One of the directories is empty!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = min(50, len(fake_files))
    fid_value = fid_score.calculate_fid_given_paths(
        [temp_real, temp_fake],
        batch_size, device, dims,
        num_workers=0
    )
    return fid_value

def main():
    real_dir = "../dermDatabaseOfficial/release_v0/images"  
    fake_dir = "../synthetic_dataset"             

    print("\n🔍 Evaluating FID between real and synthetic images...\n")
    fid_value = compute_fid(real_dir, fake_dir)
    print(f"📊 FID Score: {fid_value:.4f}  (lower is better)")

    if fid_value < 50:
        print("✅ Generated images are considered high quality (FID < 50).")
    else:
        print("⚠️ Generated images may need improvement (FID >= 50).")

    print("\nℹ️ SSIM evaluation skipped due to format variability. FID is used as the primary metric for realism.\n")

if __name__ == '__main__':
    main()