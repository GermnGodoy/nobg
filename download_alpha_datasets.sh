#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO = "OneOverZero/MAGICK"
TARGET_BYTES = 35_000_000_000
TEST_FRACTION = 0.1
WORKERS = 16
ROOT = Path("images")
MANIFEST = ROOT / "magick_files.txt"

ROOT.mkdir(exist_ok=True)

# Load or generate the file selection.
if MANIFEST.exists():
    files = MANIFEST.read_text().splitlines()
else:
    print("Listing MAGICK files...")

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

random.Random(123).shuffle(files)
test_count = round(len(files) * TEST_FRACTION)

def process(item):
    i, path = item
    split = "MAGICK-TE" if i < test_count else "MAGICK-TR"
    name = Path(path).name

    image_out = ROOT / split / "images" / name
    alpha_out = ROOT / split / "alphas" / name

    if image_out.exists() and alpha_out.exists():
        return

    image_out.parent.mkdir(parents=True, exist_ok=True)
    alpha_out.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=path
    )

    with Image.open(downloaded) as img:
        rgba = img.convert("RGBA")
        rgba.convert("RGB").save(image_out)
        rgba.getchannel("A").save(alpha_out)

# Download and process images concurrently.
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    tasks = [executor.submit(process, item) for item in enumerate(files)]

    with tqdm(total=len(tasks), desc="Processing MAGICK") as progress:
        for task in as_completed(tasks):
            task.result()
            progress.update(1)

print("Done.")
PY