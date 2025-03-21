import os
from glob import glob 

dataset_paths = [
    "../datasets/ISIC-images",
    "../dermDatabaseOfficial/release_v0/images"
]

print("Checking dataset folders...\n")

for path in dataset_paths:
    # Find all image files with common extensions recursively
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(glob(os.path.join(path, '**', ext), recursive=True))  # Recursive search

    print(f"Path: {path}")
    print(f"  Found {len(images)} images")
    if len(images) > 0:
        print(f"  Example file: {images[0]}")
    else:
        print("  ⚠️ No images found. Double-check the folder or path.")
    print()

print(" Done checking.")
