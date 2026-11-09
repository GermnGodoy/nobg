"""
Data transforms for the dataset. (made following albumentations library documentation)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

# ---------------------------------------------
# Data Augmentation

def letterbox_geometry(height, width, image_size):
    """Match LongestMaxSize + centered PadIfNeeded, also for undoing CLI padding."""
    scale = image_size / max(height, width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    return (resized_height, resized_width,
            (image_size - resized_height) // 2, (image_size - resized_width) // 2)


def _resize_transforms(image_size, preserve_aspect_ratio):
    if preserve_aspect_ratio:
        return [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        ]
    return [A.Resize(image_size, image_size)]


class Augmenter:
    """
    A class to perform data augmentation on images and masks.
    """
    
    def __init__(self, image_size, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, 
                 crop_prob=0.5, brightness_contrast_prob=0.5, rotate_limit=45,
                 crop_before_resize=False, crop_scale=(0.08, 1.0), seed=None, matting=False,
                 preserve_aspect_ratio=False):

        resize = _resize_transforms(image_size, preserve_aspect_ratio)
        
        crop = A.RandomResizedCrop(
            size=(image_size, image_size), scale=tuple(crop_scale), p=crop_prob,
            ratio=(1.0, 1.0) if preserve_aspect_ratio else (0.75, 4 / 3),
        )

        if crop_before_resize or preserve_aspect_ratio:
            resize.insert(0, crop)
            geometry = A.Rotate(limit=rotate_limit, p=rotate_prob)
        
        else:
            geometry = A.OneOf([
                A.Rotate(limit=rotate_limit, p=rotate_prob),
                crop], p=0.5)

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
        

def get_val_augmentation(image_size, matting=False, preserve_aspect_ratio=False):
    """
    Returns the validation augmentation pipeline.
    """
    return A.Compose([
        *_resize_transforms(image_size, preserve_aspect_ratio),
        A.Normalize(),
        ToTensorV2()
    ], mask_interpolation=cv2.INTER_LINEAR if matting else cv2.INTER_NEAREST)
