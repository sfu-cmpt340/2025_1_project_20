import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load both datasets
real_df = pd.read_csv("./extracted_features_with_labels.csv")
real_df["source"] = "real"

synthetic_df = pd.read_csv("./synthetic_features.csv")
synthetic_df["source"] = "synthetic"

##please uncomment the lines 14 and 15 when running in train_feature directory and comment lines 7 and 10
# real_df = pd.read_csv("../extracted_features_with_labels.csv")
# synthetic_df = pd.read_csv("../synthetic_features.csv")


# Combine and isolate features
combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
features = combined_df.drop(columns=["filename", "label", "source"])

# Reduce dimensions
print("🎨 Running PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
combined_df["PC1"] = pca_result[:, 0]
combined_df["PC2"] = pca_result[:, 1]

# Plot by label
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x="PC1", y="PC2", hue="label", palette="tab20", s=40)
plt.title("PCA: Real + Synthetic by Class")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot by source
plt.figure(figsize=(8, 6))
sns.scatterplot(data=combined_df, x="PC1", y="PC2", hue="source", palette="Set1", s=60)
plt.title("PCA: Real vs Synthetic")
plt.tight_layout()
plt.show()
