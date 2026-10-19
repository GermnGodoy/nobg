#!/usr/bin/env bash
set -euo pipefail

# Official dataset: https://saliencydetection.net/duts/
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="${PROJECT_DIR}/images"
BASE_URL="https://saliencydetection.net/duts/download"

for program in curl unzip; do
    if ! command -v "$program" >/dev/null 2>&1; then
        printf 'Error: %s is required to download the datasets.\n' "$program" >&2
        exit 1
    fi
done

mkdir -p "$IMAGES_DIR"

dataset_complete() {
    local dataset="$1" expected="$2"
    local image_dir="${IMAGES_DIR}/${dataset}/${dataset}-Image"
    local mask_dir="${IMAGES_DIR}/${dataset}/${dataset}-Mask"

    [[ -d "$image_dir" && -d "$mask_dir" ]] || return 1
    [[ "$(find "$image_dir" -type f -name '*.jpg' | wc -l)" -eq "$expected" &&
       "$(find "$mask_dir" -type f -name '*.png' | wc -l)" -eq "$expected" ]]
}

download_dataset() {
    local dataset="$1" expected="$2"
    local archive="${IMAGES_DIR}/${dataset}.zip"

    if dataset_complete "$dataset" "$expected"; then
        printf '%s is already complete; skipping.\n' "$dataset"
        return
    fi

    if [[ ! -f "$archive" ]]; then
        printf 'Downloading %s...\n' "$dataset"
        curl --fail --location --retry 3 --connect-timeout 30 \
            --continue-at - --output "${archive}.part" \
            "${BASE_URL}/${dataset}.zip"
        mv -- "${archive}.part" "$archive"
    fi

    if ! unzip -tq "$archive" >/dev/null; then
        printf 'Error: %s is not a valid ZIP. Remove it and rerun this script.\n' "$archive" >&2
        exit 1
    fi

    printf 'Extracting %s...\n' "$dataset"
    # Preserve existing files when filling an incomplete dataset directory.
    unzip -q -n "$archive" -d "$IMAGES_DIR"

    if ! dataset_complete "$dataset" "$expected"; then
        printf 'Error: %s must contain %s JPG images and %s PNG masks.\n' \
            "$dataset" "$expected" "$expected" >&2
        exit 1
    fi
    printf '%s is ready.\n' "$dataset"
}

download_dataset DUTS-TR 10553
download_dataset DUTS-TE 5019

printf 'Datasets are available in %s\n' "$IMAGES_DIR"
