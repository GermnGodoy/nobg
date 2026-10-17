import torch

# ---------------------------------------------
# Dataset Class

class Dataset(torch.utils.data.Dataset):
    """
    A class to represent a dataset.
    """
    
    def __init__(self, mask_path, image_path, transform=None):
        self.mask_path = mask_path
        self.image_path = image_path
        self.transform = transform

    __len__ = lambda self: min(len(self.mask_path), len(self.image_path))

    def __getitem__(self, idx):
        mask = self.mask_path[idx]
        image = self.image_path[idx]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
