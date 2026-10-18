import sys
from unittest.main import main

from data.data import Dataset
from data.transforms import Augmenter, get_val_augmentation

import torch
from model.model import build_segformer

from trainer.loss import dice_loss, metrics
from trainer.optimizer import optimizers
from trainer.sampler import sample_images

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
# Train Epoch

def train_epoch(model, train_loader, device, steps, optimizer_name, loss_fn, learning_rate=1e-3):
    model.train()
    optimizer = optimizers[optimizer_name](lr=learning_rate, params=model.parameters())
    
    total_loss = 0.0
    current_step = 0
    total_steps = len(train_loader)
    if steps is not None:
        total_steps = min(total_steps, steps)

    print("===============================================")
    
    for step, (images, masks) in enumerate(train_loader):
        if steps is not None and step >= steps:
            break
        
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        current_step = step + 1
        if current_step == 1 or current_step % 10 == 0 or current_step == total_steps:
            print(
                f"Step {current_step:>3}/{total_steps} "
                f"| Mean Loss: {total_loss / current_step:.4f}",
                flush=True,
            )
    
    return total_loss / max(current_step, 1)


# ---------------------------------------------
# Validate Epoch

def validate_epoch(model, val_loader, device, loss_fn, metric_names):
    model.eval()
    total_loss = 0.0
    metric_sums = {name: 0.0 for name in metric_names}
    
    with torch.no_grad():
        for step, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            for name in metric_names:
                metric_fn = metrics[name]
                metric_value = metric_fn(outputs, masks)
                metric_sums[name] += metric_value.item()
    
    avg_loss = total_loss / (step + 1)
    avg_metrics = {name: metric_sums[name] / (step + 1) for name in metric_names}
    
    return avg_loss, avg_metrics


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

    print(f"\n===============================================")
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"===============================================\n")

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

    print("\nStarting first epoch...\n")
    print("===============================================\n")

    # Start training
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, device, steps, config["training"]["optimizer"], dice_loss, learning_rate=config["training"]["learning_rate"])
        
        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}\n")
        
        validate_loss, validate_metrics = validate_epoch(model, val_loader, device, dice_loss, config["training"]["metrics"])

        print(f"Validation Loss: {validate_loss:.4f}")
        print("Validation Metrics:")
        for name, value in validate_metrics.items():
            print(f"\t{name}: {value:.4f}\n")

        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
            print(f"Model saved at epoch {epoch + 1}")

            if config["sample_images"]:
                sample_images(model, val_loader, device, epoch + 1, config["sample_images_dir"])
                print(f"Sample images saved at epoch {epoch + 1}")
            print()

if __name__ == "__main__":  
    main()
