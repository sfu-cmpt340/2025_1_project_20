# run_all.py: One-click pipeline for DermSynth3D

import os

# Step 1: Generate synthetic images for weak classes
print("\n🧪 [1/6] Generating synthetic images...")
os.system("python3 train_feature/generate_synthetic.py")

# Step 2: Extract features from real and synthetic images
print("\n🔍 [2/6] Extracting features from real and synthetic datasets...")
os.system("python3 feature_extraction.py")

# Step 3: Train and evaluate model using real, synthetic, and combined features
print("\n🚀 [3/6] Training classifier with extracted features...")
os.system("python3 train_feature/train_combined_features.py")

# Step 4: Visualize real vs synthetic features using PCA
print("\n🎨 [4/6] Visualizing feature space via PCA...")
os.system("python3 train_feature/pca_visualization.py")

# Step 5: Evaluate synthetic image quality using FID
print("\n📊 [5/6] Evaluating FID score...")
os.system("python3 evaluate_quality.py")

# Step 6: Done!
print("\n✅ [6/6] DermSynth3D pipeline complete!")
