import os
import argparse
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

INPUT_DIR = "preprocess_images"
OUTPUT_DIR = "processed_images"
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


# ---------------------------------------------------------------------------
# Subcommand: resize
# ---------------------------------------------------------------------------

def cmd_resize(args):
    target_w, target_h = args.width, args.height
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = get_image_files(INPUT_DIR)
    if not files:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    print(f"Resizing {len(files)} image(s) to {target_w}x{target_h} ...")

    for filename in files:
        img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
        resized = img.resize((target_w, target_h), Image.LANCZOS)
        out_name = f"{stem(filename)}.png"
        resized.save(os.path.join(OUTPUT_DIR, out_name))
        print(f"  {filename}  {img.size[0]}x{img.size[1]} -> {target_w}x{target_h}")

    print("Done.")


# ---------------------------------------------------------------------------
# Subcommand: augment
# ---------------------------------------------------------------------------

def cmd_augment(args):
    target_w, target_h = args.width, args.height
    count = args.count
    seed = args.seed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            aug_img.save(os.path.join(OUTPUT_DIR, f"{name}_aug{i}.png"))

        print(f"  {filename} -> {count} augmented copies")

    total = len(files) * count
    print(f"Done. {total} augmented images written to '{OUTPUT_DIR}/'.")


# ---------------------------------------------------------------------------
# Subcommand: rotate
# ---------------------------------------------------------------------------

def cmd_rotate(args):
    target_w, target_h = args.width, args.height
    angle = args.angle
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        rotated.save(os.path.join(OUTPUT_DIR, out_name))
        print(f"  {filename} -> {out_name}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Image preprocessing pipeline for segmentation datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # resize
    p_resize = subparsers.add_parser("resize", help="Resize images to target dimensions.")
    p_resize.add_argument("width", type=int, help="Target width in pixels")
    p_resize.add_argument("height", type=int, help="Target height in pixels")
    p_resize.set_defaults(func=cmd_resize)

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
