import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from amazing.data_loader import get_dataloader
from skimage.feature import local_binary_pattern
import pandas as pd

# Parameters for LBP
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# Gabor filter setup
def build_gabor_filters():
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in (1, 3):
            for frequency in (0.1, 0.3):
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 1/frequency, 0.5, 0, ktype=cv2.CV_32F)
                filters.append(kernel)
    return filters

def apply_gabor_filters(image, filters):
    responses = []
    for kernel in filters:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        responses.append(np.mean(filtered))
    return responses

# Load pretrained ResNet50 (feature extractor mode)
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.eval()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification head

# Define transform for deep feature extraction
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load data
DATA_DIRS = [
    "../datasets/ISIC-images",
    "../dermDatabaseOfficial/release_v0/images"
]
train_loader = get_dataloader(DATA_DIRS, batch_size=1, augment=False)

gabor_filters = build_gabor_filters()

features_list = []

for i, batch in enumerate(train_loader):
    # Check if batch is a dict or tuple. If it’s a tuple: (img, label)
    img = batch[0]  # Image array

    # Convert image from (C, H, W) if necessary
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.moveaxis(img, 0, -1)  # Convert from (C, H, W) to (H, W, C)
    
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # RGB histograms
    r_hist = cv2.calcHist([img_bgr], [2], None, [256], [0, 256]).flatten()
    g_hist = cv2.calcHist([img_bgr], [1], None, [256], [0, 256]).flatten()
    b_hist = cv2.calcHist([img_bgr], [0], None, [256], [0, 256]).flatten()

    # LBP features
    lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()

    # Gabor features
    gabor_features = apply_gabor_filters(img_gray, gabor_filters)

    # Deep features
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        deep_features = resnet(img_tensor).squeeze().numpy()

    # Combine all features
    combined_features = np.concatenate([r_hist, g_hist, b_hist, lbp_hist, gabor_features, deep_features])
    features_list.append(combined_features)

    # Print debug info for every 50 images
    if i % 50 == 0:
        print(f"Processed {i} images")
        print(f"Sample feature vector length: {len(combined_features)}")
        print(f"First 10 feature values: {combined_features[:10]}")

# Save all features to CSV
features_array = np.array(features_list)
df = pd.DataFrame(features_array)
df.to_csv("extracted_features.csv", index=False)

print("✅ Feature extraction complete! Saved as extracted_features.csv")
print(f"Shape of feature matrix: {features_array.shape}")
