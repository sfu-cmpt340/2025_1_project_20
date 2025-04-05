import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load real and synthetic features

## the file location has been used to implement run.py
print("📥 Loading real and synthetic features...")
real_df = pd.read_csv("./extracted_features_with_labels.csv")
synthetic_df = pd.read_csv("./synthetic_features.csv")

##please uncomment the lines 14 and 15 when running in train_feature directory and comment 10 and 11
# real_df = pd.read_csv("../extracted_features_with_labels.csv")
# synthetic_df = pd.read_csv("../synthetic_features.csv")

combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

# Prepare features and labels
X = combined_df.drop(columns=["filename", "label"])
y = combined_df["label"]

# Split dataset
print("📊 Splitting combined dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
print("🚀 Training model on combined dataset...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("🔍 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}\n")
print("📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, xticklabels=model.classes_, 
            yticklabels=model.classes_, cmap="Blues", fmt="d")
# 🧾 Customize font sizes
plt.title("Confusion Matrix - Combined Dataset", fontsize=20)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
