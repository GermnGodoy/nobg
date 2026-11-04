"""
Data transforms for the dataset. (made following albumentations library documentation)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

# ---------------------------------------------
# Data Augmentation

class Augmenter:
    """
    A class to perform data augmentation on images and masks.
    """
    
    def __init__(self, image_size, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, 
                 crop_prob=0.5, brightness_contrast_prob=0.5, rotate_limit=45,
                 crop_before_resize=False, crop_scale=(0.08, 1.0), seed=None, matting=False):
        
        resize = [A.Resize(image_size, image_size)]

        if crop_before_resize:
            resize.insert(0, A.RandomResizedCrop(size=(image_size, image_size), scale=tuple(crop_scale), p=crop_prob))
            geometry = A.Rotate(limit=rotate_limit, p=rotate_prob)
        
        else:
            geometry = A.OneOf([
                A.Rotate(limit=rotate_limit, p=rotate_prob),
                A.RandomResizedCrop(
                    size=(image_size, image_size), scale=tuple(crop_scale), p=crop_prob,
                )], p=0.5)

        self.transform = A.Compose([
            *resize,
            A.HorizontalFlip(p=hflip_prob),
            A.VerticalFlip(p=vflip_prob),
            A.RandomBrightnessContrast(p=brightness_contrast_prob, brightness_limit=0.2, contrast_limit=0.2),
            geometry,
            A.Normalize(),
            ToTensorV2()
        ], mask_interpolation=cv2.INTER_LINEAR if matting else cv2.INTER_NEAREST, seed=seed)
    
    def __call__(self, *, image, mask):
        return self.transform(image=image, mask=mask)
        

def get_val_augmentation(image_size, matting=False):
    """
    Returns the validation augmentation pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2()
    ], mask_interpolation=cv2.INTER_LINEAR if matting else cv2.INTER_NEAREST)
