import os
import matplotlib.pyplot as plt
from PIL import Image

# Set the correct dataset path
dataset_path = "./dermDatabaseOfficial/release_v0/images/A1l/"
image_files = os.listdir(dataset_path)

# Filter out only image files (JPG, JPEG, PNG)
image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Check if images are found
if not image_files:
    print("No images found in the specified folder!")
else:
    fig, axes = plt.subplots(1, min(5, len(image_files)), figsize=(15, 5))

    for ax, img_file in zip(axes, image_files[:5]):
        img = Image.open(os.path.join(dataset_path, img_file))
        ax.imshow(img)
        ax.set_title(img_file)  # Show image filename
        ax.axis("off")  # Hide axes

    plt.show()
