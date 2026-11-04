import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------
# Sample Images

def sample_images(model, val_loader, device, epoch, output_dir="./runs/sample", dpi=100):
    """Sample one image and save it to disk for visualization."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    
    model.eval()
    with torch.no_grad():

        for i, (images, masks) in enumerate(val_loader): # we will only take the first
            images = images[:1].to(device)
            use_refiner = getattr(model, "use_refiner", False)
            if use_refiner:
                outputs, original_logits = model(images, return_original=True)
                original_alpha = original_logits.sigmoid()[0, 0].cpu().numpy()
            else:
                outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            img = images[0].cpu().permute(1, 2, 0).numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            img = np.clip(img * std + mean, 0, 1)[..., ::-1]
            mask = masks[0, 0].cpu().numpy()
            alpha = preds[0, 0].cpu().numpy()
            pred = np.dstack((img, alpha))
            
            columns = 5 if use_refiner else 4
            plt.figure(figsize=(4 * columns, 4))
            plt.subplot(1, columns, 1)
            plt.imshow(img)
            plt.title("Image")
            plt.axis('off')
            
            plt.subplot(1, columns, 2)
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
            plt.title("Target alpha")
            plt.axis('off')
            
            plt.subplot(1, columns, 3)
            plt.imshow(alpha, cmap='gray', vmin=0, vmax=1)
            plt.title("Predicted alpha")
            plt.axis('off')

            if use_refiner:
                plt.subplot(1, columns, 4)
                plt.imshow(np.dstack((img, original_alpha)))
                plt.title("SegFormer")
                plt.axis('off')

            plt.subplot(1, columns, columns)
            plt.imshow(pred)
            plt.title("Refined" if use_refiner else "Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_epoch_{epoch}_batch_{i}.png", dpi=dpi)
            plt.close()
            
            break

    model.train(was_training)
