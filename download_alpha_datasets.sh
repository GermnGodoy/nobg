
#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "OneOverZero/MAGICK"
TARGET_BYTES = 35_000_000_000
TEST_FRACTION = 0.1
ROOT = Path("images")
MANIFEST = ROOT / "magick_files.txt"

# Reuse the same selection when restarting.
ROOT.mkdir(exist_ok=True)

if MANIFEST.exists():
    files = MANIFEST.read_text().splitlines()
else:
    print("Listing MAGICK files...", flush=True)

    entries = [
        (f.path, f.size)
        for f in HfApi().list_repo_tree(
            REPO,
            path_in_repo="images",
            repo_type="dataset",
            recursive=True
        )
        if f.path.endswith(".png") and f.size
    ]

    random.Random(42).shuffle(entries)

    files, total = [], 0
    for path, size in entries:
        if total + size <= TARGET_BYTES:
            files.append(path)
            total += size

    MANIFEST.write_text("\n".join(files) + "\n")
    print(f"Selected {len(files)} images ({total / 1e9:.2f} GB)")

# Reproducible train/test split.
random.Random(123).shuffle(files)
test_count = round(len(files) * TEST_FRACTION)

for i, path in enumerate(files):
    split = "MAGICK-TE" if i < test_count else "MAGICK-TR"
    name = Path(path).stem + ".png"

    image_out = ROOT / split / "images" / name
    alpha_out = ROOT / split / "alphas" / name

    if image_out.exists() and alpha_out.exists():
        continue

    image_out.parent.mkdir(parents=True, exist_ok=True)
    alpha_out.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=path
    )

    with Image.open(downloaded) as rgba:
        rgba = rgba.convert("RGBA")
        rgba.convert("RGB").save(image_out)
        rgba.getchannel("A").save(alpha_out)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(files)}", flush=True)

print("Done.")
PY
