# Image Resize & Augmentation Pipeline

Preprocessing CLI for segmentation datasets. Resize, tile, crop ROI, rotate, and augment images using only Pillow + NumPy.

## Folder Structure

```
preprocess_images/              # input images
processed_images_resize/        # output from resize
processed_images_tile/          # output from tile
processed_images_roi/           # output from roi
processed_images_augment/       # output from augment
processed_images_rotate/        # output from rotate
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

## Installation

```bash
pip install -r requirements.txt
```

## Commands

### Tile (Strategy 1 – Best for preserving detail)

Slice large images into overlapping patches at native resolution. No downsampling — the model trains on uncompressed pixels.

```bash
python resize.py tile <patch_width> <patch_height>
python resize.py tile 1024 1024 --overlap 0.15
python resize.py tile 1024 1024 --overlap 0.2 --compress-level 1 --workers 8
```

- `--overlap` — fraction of overlap between adjacent tiles, 0.0–0.5 (default: 0.15)
- `--compress-level` — PNG compression level 0–9, lower is faster (default: 1). All levels are lossless
- `--workers` — number of parallel save threads (default: 4)
- Edge patches are automatically included so no pixels are lost
- Output filenames: `{name}_tile{i}.png`

### ROI Crop (Strategy 2 – Reduce background, keep detail)

Crop to a region of interest (manually or auto-detected), then resize. Useful when the subject occupies a small portion of the frame.

```bash
python resize.py roi <width> <height>
python resize.py roi 1024 1024 --box 500 500 2000 2000
python resize.py roi 1024 1024 --interpolation bicubic
```

- `--box X Y W H` — manual ROI coordinates. If omitted, foreground is auto-detected from corner background color
- `--interpolation` — `lanczos` (default) or `bicubic`
- Output filenames: `{name}_roi.png`

### Resize (Strategy 3 – Full-image resize)

Resize all images to target dimensions with explicit high-quality interpolation (never nearest-neighbor).

```bash
python resize.py resize <width> <height>
python resize.py resize 1024 1024 --interpolation lanczos
```

- `--interpolation` — `lanczos` (default) or `bicubic`

### Rotate

Rotate all images by a fixed angle and resize to target dimensions.

```bash
python resize.py rotate <width> <height> <angle>
python resize.py rotate 512 512 30
python resize.py rotate 512 512 -15
```

Positive angles = counter-clockwise. Output filenames: `{name}_rot{angle}.png`

### Augment

Generate multiple augmented copies per image with random combinations of transforms. All outputs are resized to target dimensions.

```bash
python resize.py augment <width> <height> --count 5
python resize.py augment <width> <height> --count 10 --seed 42
```

| Transform | Type |
|---|---|
| Random rotation (15-45 deg, positive or negative) | Spatial |
| Brightness jitter (+-30%) | Color |
| Contrast jitter (+-30%) | Color |
| Saturation jitter (+-30%) | Color |
| Gaussian noise (sigma 0-15) | Noise |
| Gaussian blur (radius 0-2) | Noise |

Each copy gets a random subset of these transforms. Output filenames: `{name}_aug{i}.png`
