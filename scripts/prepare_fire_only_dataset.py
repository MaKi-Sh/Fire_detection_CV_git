"""
Prepare a dataset with only ControlledFire and Fire classes.

Original classes:
    0: ControlledFire -> 0 (keep)
    1: Entire Image Nonfire -> delete
    2: Fire -> 1 (remap)
    3: Nonfire -> delete
    4: Smoke -> delete
"""
import os
import shutil
from pathlib import Path

# Paths
SOURCE_DIR = Path('/home/makish/Desktop/Science_fair_2025/Annotations_backup')
DEST_DIR = Path('/home/makish/Desktop/Science_fair_2025/Annotations_fire_only')

# Class mapping: old_class -> new_class (None means delete)
CLASS_MAPPING = {
    0: 0,     # ControlledFire -> 0
    1: None,  # Entire Image Nonfire -> delete
    2: 1,     # Fire -> 1
    3: None,  # Nonfire -> delete
    4: None,  # Smoke -> delete
}

NEW_CLASSES = ['ControlledFire', 'Fire']


def process_label_file(src_path: Path, dst_path: Path) -> bool:
    """
    Process a single label file, keeping only ControlledFire and Fire classes.

    Returns True if the file has any valid labels, False otherwise.
    """
    new_lines = []

    with open(src_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            old_class = int(parts[0])
            new_class = CLASS_MAPPING.get(old_class)

            if new_class is not None:
                # Remap class and keep the rest of the line
                parts[0] = str(new_class)
                new_lines.append(' '.join(parts))

    # Write the new label file (even if empty, to maintain image-label pairing)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, 'w') as f:
        f.write('\n'.join(new_lines))
        if new_lines:
            f.write('\n')

    return len(new_lines) > 0


def copy_images(src_dir: Path, dst_dir: Path):
    """Copy all images from source to destination."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for img_file in src_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            shutil.copy2(img_file, dst_dir / img_file.name)


def main():
    print("="*60)
    print("Preparing Fire-Only Dataset")
    print("="*60)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"\nNew classes: {NEW_CLASSES}")
    print(f"Class mapping: {CLASS_MAPPING}")

    # Create destination directory
    if DEST_DIR.exists():
        print(f"\nRemoving existing destination directory...")
        shutil.rmtree(DEST_DIR)

    DEST_DIR.mkdir(parents=True)

    # Process each split (train, val, test)
    splits = ['train', 'val', 'test']
    stats = {split: {'total': 0, 'with_labels': 0, 'labels_kept': 0, 'labels_removed': 0}
             for split in splits}

    for split in splits:
        print(f"\n--- Processing {split} split ---")

        src_labels_dir = SOURCE_DIR / 'labels' / split
        src_images_dir = SOURCE_DIR / 'images' / split
        dst_labels_dir = DEST_DIR / 'labels' / split
        dst_images_dir = DEST_DIR / 'images' / split

        if not src_labels_dir.exists():
            print(f"  Labels directory not found: {src_labels_dir}")
            continue

        # Process labels
        label_files = list(src_labels_dir.glob('*.txt'))
        stats[split]['total'] = len(label_files)

        for label_file in label_files:
            # Count original labels
            with open(label_file, 'r') as f:
                original_lines = [l.strip() for l in f if l.strip()]

            dst_label_path = dst_labels_dir / label_file.name
            has_labels = process_label_file(label_file, dst_label_path)

            # Count new labels
            with open(dst_label_path, 'r') as f:
                new_lines = [l.strip() for l in f if l.strip()]

            stats[split]['labels_kept'] += len(new_lines)
            stats[split]['labels_removed'] += len(original_lines) - len(new_lines)

            if has_labels:
                stats[split]['with_labels'] += 1

        print(f"  Processed {stats[split]['total']} label files")
        print(f"  Files with valid labels: {stats[split]['with_labels']}")
        print(f"  Labels kept: {stats[split]['labels_kept']}")
        print(f"  Labels removed: {stats[split]['labels_removed']}")

        # Copy images
        if src_images_dir.exists():
            print(f"  Copying images...")
            copy_images(src_images_dir, dst_images_dir)
            num_images = len(list(dst_images_dir.glob('*')))
            print(f"  Copied {num_images} images")

    # Create classes.txt
    classes_file = DEST_DIR / 'classes.txt'
    with open(classes_file, 'w') as f:
        f.write('\n'.join(NEW_CLASSES) + '\n')
    print(f"\nCreated classes.txt with classes: {NEW_CLASSES}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_kept = sum(s['labels_kept'] for s in stats.values())
    total_removed = sum(s['labels_removed'] for s in stats.values())
    print(f"Total labels kept: {total_kept}")
    print(f"Total labels removed: {total_removed}")
    print(f"Dataset saved to: {DEST_DIR}")

    return DEST_DIR


if __name__ == "__main__":
    main()
