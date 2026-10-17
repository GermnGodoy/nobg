import sys
from unittest.main import main

from data.data import Dataset
from data.transforms import Augmenter, get_val_augmentation

import torch
from model.model import build_segformer

import yaml

import time

# ---------------------------------------------
# Read Configuration

_CONFIG = "config/train.yaml"

def get_config():
    with open(_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------
# Main

def main():
    
    # Get config
    config = get_config()

    # Set Seed
    if "seed" in config["training"]:
        seed = config["training"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seed set to {seed}\n")

    # Set up logging
    if config["log_dir"] is not None:
        sys.stdout = open(config["log_dir"], "a")
        sys.stderr = open(config["log_dir"], "a")
        # clean the log file
        with open(config["log_dir"], "w") as f:
            f.write("")
            
        print(f"Logging to {config['log_dir']}\n")

    print(f"\n===============================")
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"===============================\n")

    # Create the model
    if config["model"] is None:
        print("No model specified in the configuration. Using default model.")
        model = build_segformer(num_classes=1, decoder_dim=256, decoder_dropout=0.1)
    else:
        print(f"Using model configuration: {config['model']}")
        model = build_segformer(**config["model"])
    
    if config["pretrained"] is not None:
        print(f"Using pretrained model: {config['pretrained']}")
        model.load_state_dict(torch.load(config["pretrained"], map_location="cpu", warning=False))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nTrainable parameters {len([p for p in model.parameters() if p.requires_grad])}")
    
    # Dataset and loading
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    steps = config["training"]["steps"]
    shuffle = config["training"]["shuffle"] if "shuffle" in config["training"] else True

    mask_path = config["mask_path"]
    image_path = config["image_path"]

    num_workers = config["training"]["num_workers"] if "num_workers" in config["training"] else 1

    augmenter = Augmenter(**config["augmentation"])
    val_aug = get_val_augmentation(config["augmentation"]["image_size"])

    train_data = Dataset(mask_path, image_path, transform=augmenter)
    val_data = Dataset(mask_path, image_path, transform=val_aug)

    val_percent = config["training"]["val_percent"] if "val_percent" in config["training"] else 0.1
    
    if val_percent >= 1.0:
        print("Invalid validation percentage.")
        print("Setting validation percentage to 10%")
        val_percent = 0.1

    train_indices, val_indices = torch.utils.data.random_split(range(len(train_data)), [1 - val_percent, val_percent], generator=torch.Generator().manual_seed(config["seed"]))

    train_dataset = torch.utils.data.Subset(train_data, train_indices)
    val_dataset = torch.utils.data.Subset(val_data, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Start training
    #for epoch in range(epochs):
    #    model.train()

    #    for step in range(steps):
    #        pass


    images, masks = next(iter(train_loader))
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")

if __name__ == "__main__":  
    main()