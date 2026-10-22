import sys
from pathlib import Path
from unittest.main import main

from src.nobg.data.data import Dataset
from src.nobg.data.transforms import Augmenter, get_val_augmentation

import torch
from src.nobg.model.model import build_segformer

from src.nobg.trainer.loss import bce_loss, dice_loss, metrics
from src.nobg.trainer.optimizer import optimizers
from src.nobg.trainer.sampler import sample_images

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

def train_epoch(model, train_loader, device, steps, optimizer, loss_fn):
    model.train()
    
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
    
    return total_loss / max(current_step, 1), current_step


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

    checkpoint_dir = Path("./runs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set Seed
    if "seed" in config["training"]:
        seed = config["training"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seed set to {seed}\n")

    # Set up logging
    if config["log_dir"] is not None:
        Path(config["log_dir"]).parent.mkdir(parents=True, exist_ok=True)
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
    
    checkpoint = None
    if config["pretrained"] is not None:
        print(f"Using pretrained model: {config['pretrained']}")
        saved_state = torch.load(config["pretrained"], map_location="cpu", weights_only=True)
        if "model_state_dict" in saved_state:
            checkpoint = saved_state
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(saved_state)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optimizers[config["training"]["optimizer"]](
        params=model.parameters(), lr=config["training"]["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=float(config["training"]["min_lr"])
    )

    start_epoch = 0
    global_step = 0
    if checkpoint is not None:
        saved_optimizer = checkpoint["config"]["training"]["optimizer"]
        if saved_optimizer != config["training"]["optimizer"]:
            raise ValueError(
                f"Checkpoint uses optimizer {saved_optimizer!r}, but the configuration "
                f"specifies {config['training']['optimizer']!r}."
            )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        print(f"Resuming after epoch {start_epoch}, global step {global_step}")

    print(f"\nTrainable parameters {len([p for p in model.parameters() if p.requires_grad])}")
    
    # Dataset and loading
    epochs = config["training"]["epochs"]
    if start_epoch >= epochs:
        print(
            f"Checkpoint already completed {start_epoch} epochs. "
            "Increase training.epochs to continue training."
        )
        return
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

    print(f"\nStarting epoch {start_epoch + 1}/{epochs}...\n")
    print("===============================================\n")

    weights = config["training"].get("loss") or {}
    dice_weight = float(weights.get("dice") or 0.0)
    bce_weight = float(weights.get("bce") or 0.0)

    if dice_weight == 0.0 and bce_weight == 0.0:
        print("No active loss weights specified. Using Dice Loss.")
        loss_fn = dice_loss
    else:
        def loss_fn(pred, target):
            return (
                dice_weight * dice_loss(pred, target)
                + bce_weight * bce_loss(pred, target)
            )

    # Start training
    for epoch in range(start_epoch, epochs):
        train_loss, epoch_steps = train_epoch(model, train_loader, device, steps, optimizer, loss_fn)
        global_step += epoch_steps
        
        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}\n")
        
        validate_loss, validate_metrics = validate_epoch(model, val_loader, device, loss_fn, config["training"]["metrics"])

        print(f"Validation Loss: {validate_loss:.4f}")
        print("Validation Metrics:")
        for name, value in validate_metrics.items():
            print(f"\t{name}: {value:.4f}")
        
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")

        if epoch % 1 == 0:
            print("\n================================================\n")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "loss": train_loss,
                "val_loss": validate_loss,
                "val_metrics": validate_metrics,
                "config": config,
                "scheduler_state_dict": scheduler.state_dict()
            }, checkpoint_dir / f"model_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved at epoch {epoch + 1}, global step {global_step}")

            if config["sample_images"]:
                sample_images(model, val_loader, device, epoch + 1, config["sample_images_dir"])
                print(f"Sample images saved at epoch {epoch + 1}")
            print()


if __name__ == "__main__":  
    main()
