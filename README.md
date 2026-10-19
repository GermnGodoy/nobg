# nobg

An image background removal tool for the terminal, currently in development.

Built from scratch in PyTorch, with a custom model based on the ideas behind [SegFormer](https://arxiv.org/abs/2105.15203).

Install dependencies from the project root:

```bash
python3 -m pip install -r requirements.txt
```

Download the training and test sets from the [official DUTS dataset](https://saliencydetection.net/duts/) (requires `curl` and `unzip`):

```bash
./download_datasets.sh
```

The script extracts images and masks into `images/DUTS-TR/` and `images/DUTS-TE/`.
It skips complete datasets, resumes interrupted downloads, and keeps the ZIP archives in `images/` for reuse.
