import torch
import cv2
from pathlib import Path

# ---------------------------------------------
# Dataset Class

class Dataset(torch.utils.data.Dataset):
    """
    A class to represent a dataset.
    """
    
    def __init__(self, mask_path, image_path, transform=None):
        self.mask_path = mask_path
        self.mask_routes = sorted(path for path in Path(self.mask_path).glob("*.png"))
        self.image_path = image_path
        self.image_routes = sorted(path for path in Path(self.image_path).glob("*.jpg"))
        self.transform = transform

        if [Path(p).stem for p in self.mask_routes] != [Path(p).stem for p in self.image_routes]:
            raise ValueError("Mask and image filenames do not match.")

    __len__ = lambda self: min(len(self.mask_routes), len(self.image_routes))

    def __getitem__(self, idx):
        mask = cv2.imread(str(self.mask_routes[idx]), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(self.image_routes[idx]), cv2.IMREAD_COLOR)

        if self.transform:
            transformed = self.transform(image, mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0)  # Add channel dimension for mask
        mask = mask.float() / 255.0  # Normalize mask to [0, 1]

        return image, mask
