import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from glob import glob
from torchvision import utils

# Just imported the Data Loader (from src/amazing/data_loader.py)
class DermDatasetLoader:
    def __init__(self, dataset_paths, img_size=(256, 256)):
        self.dataset_paths = dataset_paths
        self.img_size = img_size
        self.image_paths = self._gather_image_paths()

    def _gather_image_paths(self):
        image_paths = []
        for path in self.dataset_paths:
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                image_paths.extend(glob(os.path.join(path, '**', ext), recursive=True))
        return image_paths

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image at {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize to [0,1]
        return img

    def augment_image(self, image):
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        # Random rotation by 90 degrees
        if random.random() > 0.5:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # Add slight noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)
        return image

# PyTorch Dataset Wrapper using DermDatasetLoader
class DermDataset(Dataset):
    def __init__(self, dataset_paths, img_size=(256,256), augment=False):
        self.loader = DermDatasetLoader(dataset_paths, img_size)
        self.augment = augment
        self.image_paths = self.loader.image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = self.loader.load_image(image_path)
        if img is None:
            # Return a zero array if the image fails to load
            img = np.zeros((self.loader.img_size[1], self.loader.img_size[0], 3), dtype=np.float32)
        if self.augment:
            img = self.loader.augment_image(img)
        # Convert to torch tensor and change shape from (H,W,C) to (C,H,W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # Scale pixel values from [0,1] to [-1,1] to match DCGAN's Tanh output
        img = img * 2 - 1
        return img

# Updated DCGAN Models for 64×64 Images
class Generator(nn.Module):
    def __init__(self, z_dim, channels, feature_maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 4×4
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 4×4 -> 8×8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 8×8
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 8×8 -> 16×16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 16×16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 16×16 -> 32×32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),  # 32×32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 32×32 -> 64×64
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (channels, 64, 64) -> 32×32
            nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 -> 16×16
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 -> 8×8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 -> 4×4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 -> 1×1
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Output probability in [0, 1]
        )
    
    def forward(self, x):
        return self.net(x).view(-1)

def main():
    # Hyperparameters
    batch_size = 64
    lr = 0.0002
    num_epochs = 100
    z_dim = 100
    img_size = 64  # We are now using 64×64 images
    img_channels = 3
    feature_map_gen = 64
    feature_map_disc = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader Setup
    dataset_paths = ['dermDatabaseOfficial/release_v0/images']  
    dataset = DermDataset(dataset_paths, img_size=(img_size, img_size), augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize Models
    generator = Generator(z_dim, img_channels, feature_map_gen).to(device)
    discriminator = Discriminator(img_channels, feature_map_disc).to(device)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)  
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size_curr = real_images.size(0)
            
            # Creating labels for real and fake images
            real_labels = torch.full((batch_size_curr,), 1.0, device=device)
            fake_labels = torch.full((batch_size_curr,), 0.0, device=device)
            
            # Update Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images)
            loss_real = criterion(output_real, real_labels)
            
            noise = torch.randn(batch_size_curr, z_dim, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)
            
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            
            # Update Generator
            generator.zero_grad()
            output_fake_for_G = discriminator(fake_images)
            loss_G = criterion(output_fake_for_G, real_labels)  # Generator wants fake images classified as real
            loss_G.backward()
            optimizer_G.step()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        # Save sample generated images for visual inspection
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        grid = utils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.imshow(grid.permute(1, 2, 0))
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/epoch_{epoch+1}.png")
        plt.close()

        # Save model checkpoints
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")
    
    print("Training completed!")

if __name__ == '__main__':
    main()
