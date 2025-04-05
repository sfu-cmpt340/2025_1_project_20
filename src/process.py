import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Correct relative path to the image root folder
dataset_path = "../dermDatabaseOfficial/release_v0/images"

# Recursively collect all image file paths
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            image_files.append(full_path)

# Check and display results and print whats needed
if not image_files:
    print("❌ No images found in the specified folder!")
else:
    print(f"✅ Found {len(image_files)} image(s). Showing sample...")

    # Randomly sample 5 unique images
    sample_images = random.sample(image_files, k=min(5, len(image_files)))

    # Plot them
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = Image.open(img_path)

        ax = axes[i] if len(sample_images) > 1 else axes
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path))
        ax.axis("off")

    plt.tight_layout()
    plt.show()
