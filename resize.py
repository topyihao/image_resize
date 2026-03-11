import os
import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

INPUT_DIR = "preprocess_images"
OUTPUT_DIRS = {
    "resize":  "processed_images_resize",
    "tile":    "processed_images_tile",
    "roi":     "processed_images_roi",
    "augment": "processed_images_augment",
    "rotate":  "processed_images_rotate",
}
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_files(directory):
    """Return sorted list of supported image filenames in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )


def stem(filename):
    return os.path.splitext(filename)[0]


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def random_rotation_angle():
    """Return a random angle in [-45, -15] or [15, 45]."""
    angle = random.uniform(15, 45)
    if random.random() < 0.5:
        angle = -angle
    return angle


def aug_rotation(img):
    angle = random_rotation_angle()
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def aug_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))


def aug_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))


def aug_saturation(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))


def aug_gaussian_noise(img):
    sigma = random.uniform(0, 15)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def aug_gaussian_blur(img):
    radius = random.uniform(0, 2)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


ALL_AUGMENTATIONS = [
    aug_rotation, aug_brightness, aug_contrast,
    aug_saturation, aug_gaussian_noise, aug_gaussian_blur,
]


INTERPOLATION_METHODS = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
}


# ---------------------------------------------------------------------------
# Subcommand: resize (Strategy 3 – full-image resize with explicit interp)
# ---------------------------------------------------------------------------

def cmd_resize(args):
    target_w, target_h = args.width, args.height
    interp = INTERPOLATION_METHODS[args.interpolation]
    out_dir = OUTPUT_DIRS["resize"]
    os.makedirs(out_dir, exist_ok=True)

    files = get_image_files(INPUT_DIR)
    if not files:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    print(f"Resizing {len(files)} image(s) to {target_w}x{target_h} ({args.interpolation}) ...")

    for filename in files:
        img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
        resized = img.resize((target_w, target_h), interp)
        out_name = f"{stem(filename)}.png"
        resized.save(os.path.join(out_dir, out_name))
        print(f"  {filename}  {img.size[0]}x{img.size[1]} -> {target_w}x{target_h}")

    print(f"Done. Output in '{out_dir}/'.")


# ---------------------------------------------------------------------------
# Subcommand: tile (Strategy 1 – overlapping patches at native resolution)
# ---------------------------------------------------------------------------

def _compute_tile_origins(img_size, patch_size, stride):
    """Return a list of start positions along one axis, including an edge tile if needed."""
    length, patch_len, step = img_size, patch_size, stride
    origins = list(range(0, length - patch_len + 1, step))
    # add edge tile when the last regular tile doesn't reach the end
    if origins and (origins[-1] + patch_len) < length:
        origins.append(length - patch_len)
    # handle case where image is smaller than patch
    if not origins and length >= patch_len:
        origins.append(0)
    return origins


def _save_tile(arr_slice, path, compress_level):
    Image.fromarray(arr_slice).save(path, compress_level=compress_level)


def cmd_tile(args):
    patch_w, patch_h = args.width, args.height
    overlap = args.overlap
    compress = args.compress_level
    workers = args.workers
    stride_w = int(patch_w * (1 - overlap))
    stride_h = int(patch_h * (1 - overlap))
    out_dir = OUTPUT_DIRS["tile"]
    os.makedirs(out_dir, exist_ok=True)

    files = get_image_files(INPUT_DIR)
    if not files:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    print(f"Tiling {len(files)} image(s) into {patch_w}x{patch_h} patches "
          f"(overlap {overlap:.0%}, stride {stride_w}x{stride_h}, "
          f"compress {compress}, workers {workers}) ...")

    total_patches = 0
    for filename in files:
        img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
        img_w, img_h = img.size
        arr = np.array(img)
        name = stem(filename)

        x_origins = _compute_tile_origins(img_w, patch_w, stride_w)
        y_origins = _compute_tile_origins(img_h, patch_h, stride_h)

        tiles = []
        for y in y_origins:
            for x in x_origins:
                tiles.append((x, y))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for idx, (x, y) in enumerate(tiles):
                tile_arr = arr[y:y + patch_h, x:x + patch_w]
                path = os.path.join(out_dir, f"{name}_tile{idx}.png")
                futures.append(pool.submit(_save_tile, tile_arr, path, compress))
            for f in futures:
                f.result()

        total_patches += len(tiles)
        print(f"  {filename} ({img_w}x{img_h}) -> {len(tiles)} tiles")

    print(f"Done. {total_patches} tiles written to '{out_dir}/'.")


# ---------------------------------------------------------------------------
# Subcommand: roi (Strategy 2 – crop to ROI then resize)
# ---------------------------------------------------------------------------

def _auto_detect_roi(img, threshold=10):
    """Detect bounding box of non-background content.

    Assumes the background is near-black or near-white at the image edges.
    Falls back to the full image if detection fails.
    """
    arr = np.array(img)
    # sample corner pixels to estimate background color
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    bg = np.mean(corners, axis=0)
    # mask of pixels that differ from the background
    diff = np.linalg.norm(arr.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = diff > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0, img.size[0], img.size[1]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)


def cmd_roi(args):
    target_w, target_h = args.width, args.height
    interp = INTERPOLATION_METHODS[args.interpolation]
    box = args.box
    out_dir = OUTPUT_DIRS["roi"]
    os.makedirs(out_dir, exist_ok=True)

    files = get_image_files(INPUT_DIR)
    if not files:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    mode = "manual" if box else "auto-detect"
    print(f"ROI crop ({mode}) + resize to {target_w}x{target_h} for {len(files)} image(s) ...")

    for filename in files:
        img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
        name = stem(filename)

        if box:
            x, y, w, h = box
            roi = img.crop((x, y, x + w, y + h))
            roi_desc = f"({x},{y},{w}x{h})"
        else:
            x1, y1, x2, y2 = _auto_detect_roi(img)
            roi = img.crop((x1, y1, x2, y2))
            roi_desc = f"auto({x1},{y1},{x2 - x1}x{y2 - y1})"

        resized = roi.resize((target_w, target_h), interp)
        out_name = f"{name}_roi.png"
        resized.save(os.path.join(out_dir, out_name))
        print(f"  {filename} {roi_desc} -> {target_w}x{target_h}")

    print(f"Done. Output in '{out_dir}/'.")


# ---------------------------------------------------------------------------
# Subcommand: augment
# ---------------------------------------------------------------------------

def cmd_augment(args):
    target_w, target_h = args.width, args.height
    count = args.count
    seed = args.seed
    out_dir = OUTPUT_DIRS["augment"]
    os.makedirs(out_dir, exist_ok=True)

    files = get_image_files(INPUT_DIR)
    if not files:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    print(f"Augmenting {len(files)} image(s), {count} copies each, output {target_w}x{target_h} ...")

    for filename in files:
        img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
        name = stem(filename)

        for i in range(count):
            if seed is not None:
                img_seed = hash((seed, filename, i)) & 0xFFFFFFFF
                random.seed(img_seed)
                np.random.seed(img_seed % (2**32))

            n_transforms = random.randint(1, len(ALL_AUGMENTATIONS))
            transforms = random.sample(ALL_AUGMENTATIONS, n_transforms)

            aug_img = img.copy()
            for transform in transforms:
                aug_img = transform(aug_img)

            aug_img = aug_img.resize((target_w, target_h), Image.LANCZOS)
            aug_img.save(os.path.join(out_dir, f"{name}_aug{i}.png"))

        print(f"  {filename} -> {count} augmented copies")

    total = len(files) * count
    print(f"Done. {total} augmented images written to '{out_dir}/'.")


# ---------------------------------------------------------------------------
# Subcommand: rotate
# ---------------------------------------------------------------------------

def cmd_rotate(args):
    target_w, target_h = args.width, args.height
    angle = args.angle
    out_dir = OUTPUT_DIRS["rotate"]
    os.makedirs(out_dir, exist_ok=True)

    files = get_image_files(INPUT_DIR)
    if not files:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    print(f"Rotating {len(files)} image(s) by {angle}°, output {target_w}x{target_h} ...")

    for filename in files:
        img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
        name = stem(filename)

        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
        rotated = rotated.resize((target_w, target_h), Image.LANCZOS)
        out_name = f"{name}_rot{angle}.png"
        rotated.save(os.path.join(out_dir, out_name))
        print(f"  {filename} -> {out_name}")

    print(f"Done. Output in '{out_dir}/'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Image preprocessing pipeline for segmentation datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # resize (Strategy 3: full-image resize with explicit interpolation)
    p_resize = subparsers.add_parser("resize", help="Resize full images with Lanczos/Bicubic interpolation.")
    p_resize.add_argument("width", type=int, help="Target width in pixels")
    p_resize.add_argument("height", type=int, help="Target height in pixels")
    p_resize.add_argument("--interpolation", choices=["lanczos", "bicubic"], default="lanczos",
                          help="Interpolation method (default: lanczos)")
    p_resize.set_defaults(func=cmd_resize)

    # tile (Strategy 1: overlapping patches at native resolution)
    p_tile = subparsers.add_parser("tile", help="Slice images into overlapping patches at native resolution.")
    p_tile.add_argument("width", type=int, help="Patch width in pixels (e.g. 1024)")
    p_tile.add_argument("height", type=int, help="Patch height in pixels (e.g. 1024)")
    p_tile.add_argument("--overlap", type=float, default=0.15,
                        help="Overlap fraction between tiles, 0.0-0.5 (default: 0.15)")
    p_tile.add_argument("--compress-level", type=int, default=1, choices=range(0, 10),
                        metavar="0-9", help="PNG compress level, 0=fastest 9=smallest (default: 1)")
    p_tile.add_argument("--workers", type=int, default=4,
                        help="Number of parallel save threads (default: 4)")
    p_tile.set_defaults(func=cmd_tile)

    # roi (Strategy 2: crop to ROI then resize)
    p_roi = subparsers.add_parser("roi", help="Crop to region of interest, then resize.")
    p_roi.add_argument("width", type=int, help="Target output width in pixels")
    p_roi.add_argument("height", type=int, help="Target output height in pixels")
    p_roi.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                       help="Manual ROI as x y w h. If omitted, auto-detects foreground.")
    p_roi.add_argument("--interpolation", choices=["lanczos", "bicubic"], default="lanczos",
                       help="Interpolation method (default: lanczos)")
    p_roi.set_defaults(func=cmd_roi)

    # augment
    p_aug = subparsers.add_parser("augment", help="Generate augmented copies of each image.")
    p_aug.add_argument("width", type=int, help="Target output width in pixels")
    p_aug.add_argument("height", type=int, help="Target output height in pixels")
    p_aug.add_argument("--count", type=int, default=5, help="Number of augmented copies per image (default: 5)")
    p_aug.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p_aug.set_defaults(func=cmd_augment)

    # rotate
    p_rot = subparsers.add_parser("rotate", help="Rotate all images by a fixed angle.")
    p_rot.add_argument("width", type=int, help="Target output width in pixels")
    p_rot.add_argument("height", type=int, help="Target output height in pixels")
    p_rot.add_argument("angle", type=float, help="Rotation angle in degrees (positive = counter-clockwise)")
    p_rot.set_defaults(func=cmd_rotate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
