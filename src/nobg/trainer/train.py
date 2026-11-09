import sys
from pathlib import Path

from src.nobg.data.data import Dataset
from src.nobg.data.transforms import Augmenter, get_val_augmentation

import torch
from src.nobg.model.model import build_segformer, load_segformer_weights

from src.nobg.trainer.loss import bce_loss, dice_loss, edge_loss, l1_loss, gradient_loss, detail_loss, metrics
from src.nobg.trainer.optimizer import optimizers
from src.nobg.trainer.sampler import sample_images

import yaml

import time

# ---------------------------------------------
# Read Configuration

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "train.yaml"

def get_config():
    with open(_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    return config


def restore_training_state(model, optimizer, scheduler, config, checkpoint):
    """Resume the same phase; start fresh when adding the refiner or changing phase."""
    if checkpoint is None or not config.get("resume", True):
        return 0, 0
    saved_refiner = any(key.startswith("refiner.") for key in checkpoint["model_state_dict"])
    saved_training = checkpoint["config"]["training"]
    saved_phase = saved_training.get("refiner_phase", "refiner") if saved_refiner else None
    if saved_refiner != model.use_refiner or saved_phase != model.refiner_phase:
        print("Starting a new refiner phase from loaded weights; optimizer and counters reset.")
        return 0, 0
    saved_optimizer = saved_training["optimizer"]
    if saved_optimizer != config["training"]["optimizer"]:
        raise ValueError(
            f"Checkpoint uses optimizer {saved_optimizer!r}, but the configuration "
            f"specifies {config['training']['optimizer']!r}. Set resume: false to load only weights."
        )
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]


def build_loss(training):
    weights = training.get("loss") or {}
    dice_weight = float(weights.get("dice") or 0.0)
    bce_weight = float(weights.get("bce") or 0.0)
    l1_weight = float(weights.get("l1") or 0.0)
    gradient_weight = float(weights.get("gradient") or 0.0)
    detail_weight = float(weights.get("detail") or 0.0)
    edge_weight = float(weights.get("boundary") or 0.0)
    if not any((dice_weight, bce_weight, l1_weight, gradient_weight, detail_weight)):
        print("No active base loss weights specified. Using Dice Loss.")
        dice_weight = 1.0

    def loss_fn(pred, target):
        loss = pred.new_zeros(())
        for weight, fn in ((dice_weight, dice_loss), (bce_weight, bce_loss),
                           (l1_weight, l1_loss), (gradient_weight, gradient_loss),
                           (detail_weight, detail_loss)):
            if weight:
                loss = loss + weight * fn(pred, target)
        if edge_weight:
            loss = loss + edge_weight * edge_loss(
                pred, target, kernel_size=training.get("boundary_kernel_size", 7),
            )
        return loss

    return loss_fn

# ---------------------------------------------
# Train Epoch

def train_epoch(model, train_loader, device, steps, optimizer, loss_fn, accumulation_steps=1):
    if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
        raise ValueError("accumulation_steps must be a positive integer")
    model.train()
    
    total_loss = 0.0
    current_step = 0
    total_steps = len(train_loader)
    if steps is not None:
        total_steps = min(total_steps, steps)

    print("===============================================")
    optimizer.zero_grad(set_to_none=True)
    
    for step, (images, masks) in enumerate(train_loader):
        if steps is not None and step >= steps:
            break
        
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        # Scale the final partial window too, rather than silently dropping it
        # or giving it a smaller update than the preceding windows.
        window_start = (step // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, total_steps - window_start)
        (loss / window_size).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == total_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
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

def validate_epoch(model, val_loader, device, loss_fn, metric_names, compare_original=False):
    model.eval()
    total_loss = 0.0
    metric_sums = {name: 0.0 for name in metric_names}
    compare_original = compare_original and getattr(model, "use_refiner", False)
    if compare_original:
        metric_sums.update({f"SegFormer {name}": 0.0 for name in metric_names})
    
    with torch.no_grad():
        for step, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            if compare_original:
                outputs, original = model(images, return_original=True)
            else:
                outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            for name in metric_names:
                metric_fn = metrics[name]
                metric_value = metric_fn(outputs, masks)
                metric_sums[name] += metric_value.item()
                if compare_original:
                    metric_sums[f"SegFormer {name}"] += metric_fn(original, masks).item()
    
    avg_loss = total_loss / (step + 1)
    avg_metrics = {name: total / (step + 1) for name, total in metric_sums.items()}
    
    return avg_loss, avg_metrics


# ---------------------------------------------
# Main

def main():
    
    # Get config
    config = get_config()

    checkpoint_dir = Path(config.get("checkpoint_dir", "./runs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set Seed
    seed = config.get("seed", config["training"].get("seed"))
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seed set to {seed}\n")

    # Set up logging
    if config["log_dir"] is not None:
        Path(config["log_dir"]).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(config["log_dir"], "a", buffering=1)
        sys.stderr = open(config["log_dir"], "a", buffering=1)
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
            load_segformer_weights(model, checkpoint["model_state_dict"])
        else:
            load_segformer_weights(model, saved_state)
        
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    training = config["training"]
    warmup_epochs = training.get("refiner_warmup_epochs", 0)
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError("refiner_warmup_epochs must be a non-negative integer")
    if model.use_refiner:
        target_phase = training.get("refiner_phase", "refiner")
        model.configure_refiner_training(target_phase)
        groups = [{
            "params": model.refiner.parameters(),
            "lr": float(training.get("refiner_learning_rate", 1e-4)),
        }]
        if model.refiner_phase == "decoder":
            groups.append({
                "params": model.decoder.parameters(),
                "lr": float(training.get("decoder_learning_rate", 1e-5)),
            })
        # Save the effective optimizer in the existing checkpoint configuration.
        training["optimizer"] = "adamw"
        optimizer = optimizers["adamw"](groups)
    else:
        optimizer = optimizers[training["optimizer"]](
            params=model.parameters(), lr=training["learning_rate"],
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=float(config["training"]["min_lr"])
    )

    start_epoch, global_step = restore_training_state(model, optimizer, scheduler, config, checkpoint)
    if start_epoch:
        print(f"Resuming after epoch {start_epoch}, global step {global_step}")

    print(f"\nTrainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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

    print("Preparing data augmentation...")
    augmenter = Augmenter(**config["augmentation"], seed=seed)
    val_aug = get_val_augmentation(
        config["augmentation"]["image_size"],
        matting=config["augmentation"].get("matting", False),
        preserve_aspect_ratio=config["augmentation"].get("preserve_aspect_ratio", False),
    )

    val_mask_path = config.get("val_mask_path")
    val_image_path = config.get("val_image_path")
    if bool(val_mask_path) != bool(val_image_path):
        raise ValueError("Set both val_mask_path and val_image_path, or omit both to split the training data.")

    print(f"Indexing training images: {image_path}")
    train_data = Dataset(mask_path, image_path, transform=augmenter,
                         background_path=config.get("background_path"))
    val_compositing = dict(background_path=config.get("val_background_path"),
                           deterministic_composites=True, seed=seed or 0)
    if val_mask_path and val_image_path:
        train_dataset = train_data
        print(f"Indexing validation images: {val_image_path}")
        val_dataset = Dataset(val_mask_path, val_image_path, transform=val_aug, **val_compositing)
    else:
        val_data = Dataset(mask_path, image_path, transform=val_aug, **val_compositing)
        val_percent = config["training"]["val_percent"] if "val_percent" in config["training"] else 0.1

        if val_percent >= 1.0:
            print("Invalid validation percentage.")
            print("Setting validation percentage to 10%")
            val_percent = 0.1

        train_indices, val_indices = torch.utils.data.random_split(range(len(train_data)), [1 - val_percent, val_percent], generator=torch.Generator().manual_seed(config["seed"]))

        train_dataset = torch.utils.data.Subset(train_data, train_indices)
        val_dataset = torch.utils.data.Subset(val_data, val_indices)

    print(f"Dataset ready: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\nStarting epoch {start_epoch + 1}/{epochs}...\n")
    print("===============================================\n")

    loss_fn = build_loss(config["training"])
    best_val_loss = float("inf")
    if start_epoch:
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))

    # Start training
    for epoch in range(start_epoch, epochs):
        if model.use_refiner:
            phase = "refiner" if epoch < warmup_epochs else target_phase
            model.configure_refiner_training(phase)
            print(f"Training phase: {phase} (epoch {epoch + 1})", flush=True)
        train_loss, epoch_steps = train_epoch(
            model, train_loader, device, steps, optimizer, loss_fn,
            accumulation_steps=training.get("accumulation_steps", 1),
        )
        global_step += epoch_steps
        
        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}\n")
        
        validate_loss, validate_metrics = validate_epoch(
            model, val_loader, device, loss_fn, config["training"]["metrics"],
            compare_original=config["training"].get("compare_original", False),
        )

        print(f"Validation Loss: {validate_loss:.4f}")
        print("Validation Metrics:")
        for name, value in validate_metrics.items():
            print(f"\t{name}: {value:.4f}")
        
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")

        if epoch % 1 == 0:
            print("\n================================================\n")
            is_best = validate_loss < best_val_loss
            best_val_loss = min(best_val_loss, validate_loss)
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "loss": train_loss,
                "val_loss": validate_loss,
                "val_metrics": validate_metrics,
                "best_val_loss": best_val_loss,
                "config": config,
                "scheduler_state_dict": scheduler.state_dict()
            }
            torch.save(state, checkpoint_dir / f"model_epoch_{epoch + 1}.pth")
            if is_best:
                torch.save(state, checkpoint_dir / "best.pth")
                print(f"Best validation loss improved to {best_val_loss:.6f}")
            print(f"Checkpoint saved at epoch {epoch + 1}, global step {global_step}")

            if config["sample_images"]:
                sample_images(
                    model, val_loader, device, epoch + 1, config["sample_images_dir"],
                    dpi=config.get("sample_images_dpi", 100),
                )
                print(f"Sample images saved at epoch {epoch + 1}")
            print()


if __name__ == "__main__":
    main()
