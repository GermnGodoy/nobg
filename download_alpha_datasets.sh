#!/usr/bin/env bash
set -euo pipefail

# Saves native foreground RGB + alpha; the training loader composites backgrounds.
# Example: WORKERS=32 PNG_COMPRESS_LEVEL=1 bash download_alpha_datasets.sh
python3 - <<'PY'
import os
import random
from pathlib import Path
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from PIL import Image
from tqdm import tqdm

# Keep one overall bar instead of one Hugging Face bar per download thread.
tqdm.set_lock(RLock())
disable_progress_bars()

REPO = "OneOverZero/MAGICK"
TARGET_BYTES = 35_000_000_000
TEST_FRACTION = 0.1
WORKERS = int(os.environ.get("WORKERS", "32"))
# Lower PNG compression uses less CPU without changing image/alpha pixels.
PNG_COMPRESS_LEVEL = int(os.environ.get("PNG_COMPRESS_LEVEL", "1"))
ROOT = Path("images")
MANIFEST = ROOT / "magick_files.txt"

if WORKERS < 1:
    raise SystemExit("WORKERS must be at least 1")
if not 0 <= PNG_COMPRESS_LEVEL <= 9:
    raise SystemExit("PNG_COMPRESS_LEVEL must be between 0 and 9")

ROOT.mkdir(exist_ok=True)
print(f"Workers: {WORKERS}; PNG compression: {PNG_COMPRESS_LEVEL}")

# Load or generate the file selection.
if MANIFEST.exists():
    files = MANIFEST.read_text().splitlines()
else:
    print("Listing MAGICK files...")

    entries = [
        (f.path, f.size)
        for f in tqdm(
            HfApi().list_repo_tree(REPO, path_in_repo="images", repo_type="dataset", recursive=True),
            desc="Listing MAGICK", unit="files", dynamic_ncols=True,
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
        return False

    image_out.parent.mkdir(parents=True, exist_ok=True)
    alpha_out.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=path
    )

    with Image.open(downloaded) as img:
        rgba = img.convert("RGBA")
        rgba.convert("RGB").save(image_out, compress_level=PNG_COMPRESS_LEVEL)
        rgba.getchannel("A").save(alpha_out, compress_level=PNG_COMPRESS_LEVEL)
    return True

# Download and process images concurrently.
with tqdm(total=len(files), desc="MAGICK", unit="img", dynamic_ncols=True, miniters=1) as progress:
    created = existing = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        tasks = [executor.submit(process, item) for item in enumerate(files)]
        for task in as_completed(tasks):
            if task.result():
                created += 1
            else:
                existing += 1
            progress.set_postfix(new=created, existing=existing, refresh=False)
            progress.update(1)

print("Done.")
PY
