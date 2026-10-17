import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------
# Sample Images

def sample_images(model, val_loader, device, epoch, output_dir="./runs/sample"):
    """Sample one image and save it to disk for visualization."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    
    model.eval()
    with torch.no_grad():

        for i, (images, masks) in enumerate(val_loader): # we will only take the first
            images = images[:1].to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            img = images[0].cpu().permute(1, 2, 0).numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            img = np.clip(img * std + mean, 0, 1)[..., ::-1]
            mask = masks[0, 0].cpu().numpy()
            alpha = preds[0, 0].cpu().numpy()
            pred = np.dstack((img, alpha))
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
            plt.title("Mask")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred)
            plt.title("Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_epoch_{epoch}_batch_{i}.png")
            plt.close()
            
            break

    model.train(was_training)
