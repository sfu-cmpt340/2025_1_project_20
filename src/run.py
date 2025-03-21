from amazing.data_loader import get_dataloader
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np

# Define dataset directories 
data_dirs = [
    "../datasets/ISIC-images",
    "../dermDatabaseOfficial/release_v0/images"
] 

# Get dataloader generator for testing batch loading
train_loader = get_dataloader(data_dirs, batch_size=4, augment=True)

# --- Test loading batches by printing batch shapes ---
print("Testing data loading...")
for batch in train_loader:
    print(f"Batch shape: {batch.shape}")  # Should be (batch_size, height, width, channels)
    break  

# --- Reload dataloader for visualization 
train_loader = get_dataloader(data_dirs, batch_size=4, augment=True)


try:
    batch = next(train_loader)
    print("Successfully loaded a batch for visualization.")
except StopIteration:
    print("Error: No batches found. Check dataset paths.")
    exit()

# Convert batch to tensor format for PyTorch grid visualization
batch_tensor = torch.tensor(batch).permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)

# Plot the batch images in a grid
grid = torchvision.utils.make_grid(batch_tensor, nrow=2)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis('off')
plt.title("Example Batch of Images")
plt.show()
