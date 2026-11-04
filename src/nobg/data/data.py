import torch
import cv2
import numpy as np
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
        self.image_routes = sorted(path for path in Path(self.image_path).glob("*")
                                   if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.transform = transform

        if not self.mask_routes or not self.image_routes:
            raise ValueError(f"No image/mask pairs found in {self.image_path} and {self.mask_path}.")
        if [Path(p).stem for p in self.mask_routes] != [Path(p).stem for p in self.image_routes]:
            raise ValueError("Mask and image filenames do not match.")

    __len__ = lambda self: min(len(self.mask_routes), len(self.image_routes))

    def __getitem__(self, idx):
        mask = cv2.imread(str(self.mask_routes[idx]), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(str(self.image_routes[idx]), cv2.IMREAD_COLOR)

        if mask is None or image is None:
            raise ValueError(f"Could not read image or mask: {self.image_routes[idx]}, {self.mask_routes[idx]}")
        if mask.ndim == 3:
            mask = mask[:, :, 3] if mask.shape[2] == 4 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = mask.astype(np.float32) / np.iinfo(mask.dtype).max

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = torch.as_tensor(mask).unsqueeze(0)  # [1, H, W], alpha in [0, 1]

        return image, mask
