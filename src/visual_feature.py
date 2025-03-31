import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
csv_path = "extracted_features_with_labels.csv"  # adjust as needed
output_path = "pca_visualization.png"
n_components = 2
# ---------------

# Load CSV
df = pd.read_csv(csv_path)

# Separate features and labels
X = df.drop(columns=["filename", "label"]).values
y = df["label"].values

# Run PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Wrap in a DataFrame for easier plotting
pca_df = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "Label": y
})

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Label", palette="tab10", s=60, alpha=0.8)
plt.title("PCA Visualization of Extracted Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save and show
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ PCA plot saved to: {os.path.abspath(output_path)}")
