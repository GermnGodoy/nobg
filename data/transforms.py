"""
Data transforms for the dataset. (made following albumentations library documentation)
"""

import albumentations as A
import numpy as np

# ---------------------------------------------
# Data Augmentation

class Augmenter:
    """
    A class to perform data augmentation on images and masks.
    """
    
    def __init__(self, image_size, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, 
                 crop_prob=0.5, brightness_contrast_prob=0.5, rotate_limit=45):
        self.transform = A.Compose([
            
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=hflip_prob),
            A.VerticalFlip(p=vflip_prob),
            A.RandomBrightnessContrast(p=brightness_contrast_prob, brightness_limit=0.2, contrast_limit=0.2),
            A.OneOf([
                A.RandomRotate(limit=rotate_limit, p=rotate_prob),
                A.RandomResizedCrop(p=crop_prob, height=int(image_size * 0.8), width=int(image_size * 0.8)),
            ], p=0.5),
            A.Normalize(),
        ])
    
    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']