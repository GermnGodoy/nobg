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

---

## Command-line usage

**Install** the package from the project root to register the `nobg` command:

```bash
python3 -m pip install -e .
```

Place your trained weights at `src/nobg/model/weights.pth`, or specify another weights file with `-w`:

```bash
nobg image.jpg
```

This saves a transparent PNG as `output.png` in the current directory, preserving the original image size. By default, the mask is binary with a threshold of `0.5`.

| Option | Description |
| --- | --- |
| `-w`, `--weights` | Weights file or training checkpoint. Defaults to `model/weights.pth` inside the package. |
| `-o`, `--output` | Output file with a `.png` extension. Defaults to `output.png`. |
| `-t`, `--threshold` | Binary mask threshold between `0` and `1`. Defaults to `0.5`; ignored with `--alpha`. |
| `--alpha` | Use gradual transparency from the predicted mask. |
| `-h`, `--help` | Show help and usage examples. |

```bash
nobg image.jpg -w checkpoint.pth -o result.png -t 0.7
nobg image.jpg --alpha -o result.png
nobg --help
```
