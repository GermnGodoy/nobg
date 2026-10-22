"""Command-line interface for image background removal."""

import argparse
from pathlib import Path
from pickle import UnpicklingError


def _remove_background(image_path, weights_path, output_path, alpha=False, threshold=0.5):
    # Load inference dependencies only after parsing arguments so --help works fast
    import cv2
    import numpy as np
    import torch

    from .data.transforms import get_val_augmentation
    from .model.model import build_segformer

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    saved_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    config = saved_state.get("config") or {}
    model = build_segformer(**(config.get("model") or {}))
    model.load_state_dict(saved_state.get("model_state_dict", saved_state))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    image_size = config.get("augmentation", {}).get("image_size", 256)

    inputs = get_val_augmentation(image_size)(image=image)["image"]
    inputs = inputs.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(inputs)
        if logits.shape[1] != 1:
            raise ValueError("Expected a binary segmentation model with one output channel.")
        mask = logits.sigmoid()[0, 0].cpu().numpy()

    height, width = image.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if not alpha:
        mask = mask >= threshold
    output = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    output[:, :, 3] = np.rint(mask.clip(0, 1) * 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"Could not save image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nobg",
        description="Remove an image's background and save a transparent PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nobg image.jpg\n"
            "  nobg image.jpg -w weights.pth -o result.png\n"
            "  nobg image.jpg -t 0.7\n"
            "  nobg image.jpg --alpha"
        ),
    )
    
    parser.add_argument("image", type=Path, help="Input image to remove the background from.")
    parser.add_argument(
        "-w", "--weights",
        type=Path,
        default=Path(__file__).resolve().parent / "model" / "weights.pth",
        help="Path to model weights (default: model/weights.pth inside the package).",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output.png"),
        help="Output PNG file (default: output.png).",
    )

    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Use gradual transparency instead of a binary mask.",
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Binary mask threshold between 0 and 1 (default: 0.5; ignored with --alpha).",
    )
    
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if not args.image.is_file():
        parser.error(f"Input image not found: {args.image}")
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must have a .png extension.")

    try:
        _remove_background(
            args.image, args.weights, args.output,
            alpha=args.alpha, threshold=args.threshold,
        )
    except (OSError, RuntimeError, ValueError, UnpicklingError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")

    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
