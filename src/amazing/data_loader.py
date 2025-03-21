import os
import cv2
import numpy as np
import random
from glob import glob

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
        
        # Check if the image was successfully loaded
        if img is None:
            print(f"Warning: Failed to load image at {image_path}")
            return None  # Skip this image if loading failed
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize to [0, 1]
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

    def load_dataset(self, augment=False):
        images = []
        for path in self.image_paths:
            img = self.load_image(path)
            
            if img is None:  # Skip if image loading failed
                continue
            
            if augment:
                img = self.augment_image(img)
            images.append(img)
        
        return np.array(images)

def get_dataloader(dataset_paths, img_size=(256, 256), augment=False, batch_size=4):
    loader = DermDatasetLoader(dataset_paths, img_size)
    images = loader.load_dataset(augment=augment)
    

    def batch_generator():
        for i in range(0, len(images), batch_size):
            yield images[i:i+batch_size]
    
    return batch_generator()
