import os
import torch
from pytorch_fid import fid_score
import cv2
from skimage.metrics import structural_similarity as ssim
import glob
import numpy as np

def compute_fid(real_dir, fake_dir, dims=2048):
    real_files = sorted(glob.glob(os.path.join(real_dir, "**", "*.*"), recursive=True))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.*")))

    print("Number of real images:", len(real_files))
    print("Number of fake images:", len(fake_files))

    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError("One of the directories is empty!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = min(50, len(fake_files)) if len(fake_files) > 0 else 1  
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size, device, dims
    )
    return fid_value

def compute_average_ssim(real_dir, fake_dir):
    """
    Compute the average SSIM between corresponding real and fake images.
    Assumes that the images in both directories are sorted and correspond to each other.
    """

    real_files = sorted(glob.glob(os.path.join(real_dir, "**", "*.*"), recursive=True))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.png")))

    
    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError("One of the directories is empty!")

    # Use the minimum number of images to avoid mismatches.
    num_images = min(len(real_files), len(fake_files))
    ssim_scores = []
    for i in range(num_images):
        real_img = cv2.imread(real_files[i])
        fake_img = cv2.imread(fake_files[i])
        if real_img is None or fake_img is None:
            print(f"Warning: Failed to load image pair {real_files[i]}, {fake_files[i]}")
            continue
        # Convert BGR to RGB
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
        # Compute SSIM over color images (multichannel=True)
        score, _ = ssim(real_img, fake_img, full=True, multichannel=True)
        ssim_scores.append(score)
    return np.mean(ssim_scores)



def main():
    real_dir = "dermDatabaseOfficial/release_v0/images"  
    fake_dir = "outputs"             
    
    # Compute FID
    fid_value = compute_fid(real_dir, fake_dir)
    print("FID Score:", fid_value)

    # Compute Average SSIM
    avg_ssim = compute_average_ssim(real_dir, fake_dir)
    print("Average SSIM:", avg_ssim)

    # Interpretation example:
    if fid_value < 50:
        print("Generated images are considered high quality (FID < 50).")
    else:
        print("Generated images may need improvement (FID >= 50).")
        
if __name__ == '__main__':
    main()