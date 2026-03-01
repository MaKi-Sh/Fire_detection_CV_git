"""
Visualize annotated images with bounding boxes from the fire-only dataset.
"""
import cv2
import random
from pathlib import Path

# Configuration
DATASET_DIR = Path('/home/makish/Desktop/Science_fair_2025/Annotations_fire_only')
OUTPUT_DIR = Path('/home/makish/Desktop/Science_fair_2025/annotation_samples')
NUM_SAMPLES = 100

# Class names and colors (BGR format)
CLASS_NAMES = ['ControlledFire', 'Fire']
CLASS_COLORS = [
    (0, 165, 255),   # ControlledFire - Orange
    (0, 0, 255),     # Fire - Red
]


def draw_annotations(image_path: Path, label_path: Path) -> tuple:
    """
    Draw bounding boxes on an image based on YOLO format labels.

    Returns: (annotated_image, list of class indices found)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, []

    h, w = img.shape[:2]
    classes_found = []

    # Read and draw labels
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])

                # Convert from normalized to pixel coordinates
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                # Get color and label
                color = CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)
                label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"

                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + label_size[0] + 4, y1), color, -1)

                # Draw label text
                cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                classes_found.append(class_id)

    return img, classes_found


def main():
    print("="*60)
    print("Visualizing Fire-Only Dataset Annotations")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all image-label pairs from all splits
    all_samples = []

    for split in ['train', 'val', 'test']:
        images_dir = DATASET_DIR / 'images' / split
        labels_dir = DATASET_DIR / 'labels' / split

        if not images_dir.exists():
            continue

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                label_path = labels_dir / (img_path.stem + '.txt')
                all_samples.append((img_path, label_path, split))

    print(f"\nTotal images found: {len(all_samples)}")

    # Filter to only include images with annotations
    samples_with_labels = []
    for img_path, label_path, split in all_samples:
        if label_path.exists():
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:  # Has non-empty labels
                    samples_with_labels.append((img_path, label_path, split))

    print(f"Images with labels: {len(samples_with_labels)}")

    # Select random samples (prioritize images with labels)
    num_to_select = min(NUM_SAMPLES, len(samples_with_labels))
    selected = random.sample(samples_with_labels, num_to_select)

    print(f"\nGenerating {num_to_select} visualizations...")

    # Process and save
    stats = {'ControlledFire': 0, 'Fire': 0}

    for i, (img_path, label_path, split) in enumerate(selected):
        annotated_img, classes_found = draw_annotations(img_path, label_path)

        if annotated_img is None:
            print(f"  Skipped (could not read): {img_path.name}")
            continue

        # Update stats
        for cls_id in classes_found:
            if cls_id < len(CLASS_NAMES):
                stats[CLASS_NAMES[cls_id]] += 1

        # Save with split prefix
        output_path = OUTPUT_DIR / f"{split}_{i:03d}_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_to_select}")

    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total samples generated: {num_to_select}")
    print(f"\nLabel counts in samples:")
    for cls_name, count in stats.items():
        print(f"  {cls_name}: {count}")

    print(f"\nColor legend:")
    print(f"  ControlledFire: Orange boxes")
    print(f"  Fire: Red boxes")


if __name__ == "__main__":
    main()
