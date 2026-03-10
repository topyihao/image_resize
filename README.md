# Image Resize & Augmentation Pipeline

Preprocessing CLI for segmentation datasets. Resize, rotate, and augment images using only Pillow + NumPy.

## Folder Structure

```
preprocess_images/      # input images
processed_images/       # output images
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

## Installation

```bash
pip install Pillow numpy
```

## Commands

### Resize

Resize all images to target dimensions.

```bash
python resize.py resize <width> <height>
```

### Rotate

Rotate all images by a fixed angle and resize to target dimensions.

```bash
python resize.py rotate <width> <height> <angle>
```

Example:

```bash
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
