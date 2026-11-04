import torch
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------
# Dataset Class

class Dataset(torch.utils.data.Dataset):
    """
    Image/alpha pairs matched by filename stem; unpaired or invalid files are skipped.
    """
    
    def __init__(self, mask_path, image_path, transform=None, validate_files=True):
        self.mask_path = mask_path
        self.mask_routes = sorted(path for path in Path(self.mask_path).glob("*.png"))
        self.image_path = image_path
        self.image_routes = sorted(path for path in Path(self.image_path).glob("*")
                                   if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.transform = transform

        if not self.mask_routes or not self.image_routes:
            raise ValueError(f"No image/mask pairs found in {self.image_path} and {self.mask_path}.")

        masks = {path.stem: path for path in self.mask_routes}
        images = {path.stem: path for path in self.image_routes}
        if len(images) != len(self.image_routes):
            raise ValueError(f"Multiple images share the same filename stem in {self.image_path}.")
        names = sorted(images.keys() & masks.keys())
        if not names:
            raise ValueError(f"No matching image/mask pairs found in {self.image_path} and {self.mask_path}.")

        skipped_images = len(images) - len(names)
        skipped_masks = len(masks) - len(names)
        if skipped_images or skipped_masks:
            print(f"Skipping unpaired files in {self.image_path} / {self.mask_path}: "
                  f"{skipped_images} images, {skipped_masks} alphas.")
        self.image_routes = [images[name] for name in names]
        self.mask_routes = [masks[name] for name in names]

        if validate_files:
            total = len(self.image_routes)
            print(f"Checking readability of {total} image/alpha pairs...", flush=True)
            valid_indices = []
            for idx in range(total):
                try:
                    self._read_pair(idx)
                except (ValueError, cv2.error) as error:
                    print(f"Skipping pair {self.image_routes[idx]} / {self.mask_routes[idx]}: {error}", flush=True)
                else:
                    valid_indices.append(idx)
                checked = idx + 1
                if checked % 100 == 0 or checked == total:
                    print(f"Checked {checked}/{total} pairs; skipped {checked - len(valid_indices)}.", flush=True)
            if not valid_indices:
                raise ValueError(f"No readable image/mask pairs found in {self.image_path} and {self.mask_path}.")
            self.image_routes = [self.image_routes[idx] for idx in valid_indices]
            self.mask_routes = [self.mask_routes[idx] for idx in valid_indices]

    __len__ = lambda self: min(len(self.mask_routes), len(self.image_routes))

    def _read_pair(self, idx):
        mask = cv2.imread(str(self.mask_routes[idx]), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(str(self.image_routes[idx]), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {self.image_routes[idx]}")
        if mask is None:
            raise ValueError(f"Could not read mask: {self.mask_routes[idx]}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image and mask sizes differ: {image.shape[:2]} vs {mask.shape[:2]}")
        return image, mask

    def __getitem__(self, idx):
        image, mask = self._read_pair(idx)
        if mask.ndim == 3:
            mask = mask[:, :, 3] if mask.shape[2] == 4 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = mask.astype(np.float32) / np.iinfo(mask.dtype).max

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = torch.as_tensor(mask).unsqueeze(0)  # [1, H, W], alpha in [0, 1]

        return image, mask
