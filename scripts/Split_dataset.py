from sklearn.model_selection import train_test_split
import os
import shutil

def get_category(filename):
    """Classify image as 'fire' or 'nonfire' based on filename prefix."""
    if filename.startswith('fire_') or filename.startswith('bothFireAndSmoke_'):
        return 'fire'
    return 'nonfire'

def split_dataset(images_dir, labels_dir, output_dir, test_size=0.2, val_size=0.2):
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(images_dir, f))]

    if len(images) == 0:
        print(f"No images found in {images_dir}")
        return

    # Separate fire and non-fire images
    fire_images = [img for img in images if get_category(img) == 'fire']
    nonfire_images = [img for img in images if get_category(img) == 'nonfire']

    print(f"Found {len(images)} total images")
    print(f"  - Fire images: {len(fire_images)}")
    print(f"  - Non-fire images: {len(nonfire_images)}")

    # Split each category separately to ensure balanced distribution
    fire_train, fire_test = train_test_split(fire_images, test_size=test_size, random_state=42)
    fire_train, fire_val = train_test_split(fire_train, test_size=val_size, random_state=42)

    nonfire_train, nonfire_test = train_test_split(nonfire_images, test_size=test_size, random_state=42)
    nonfire_train, nonfire_val = train_test_split(nonfire_train, test_size=val_size, random_state=42)

    # Combine back
    train_images = fire_train + nonfire_train
    val_images = fire_val + nonfire_val
    test_images = fire_test + nonfire_test

    print(f"\nStratified split:")
    print(f"  Train: {len(train_images)} ({len(fire_train)} fire, {len(nonfire_train)} non-fire)")
    print(f"  Val:   {len(val_images)} ({len(fire_val)} fire, {len(nonfire_val)} non-fire)")
    print(f"  Test:  {len(test_images)} ({len(fire_test)} fire, {len(nonfire_test)} non-fire)")

    for subset, subset_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        os.makedirs(f"{output_dir}/images/{subset}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{subset}", exist_ok=True)
        for image in subset_images:
            shutil.copy(f"{images_dir}/{image}", f"{output_dir}/images/{subset}/{image}")
            # Handle different image extensions
            label_file = os.path.splitext(image)[0] + '.txt'
            label_path = f"{labels_dir}/{label_file}"
            if os.path.exists(label_path):
                shutil.copy(label_path, f"{output_dir}/labels/{subset}/{label_file}")
            else:
                print(f"Warning: Label file not found for {image}")

    print("\nDataset split complete!")

if __name__ == "__main__":
    images_dir = "/home/makish/Desktop/Science_fair_2025/Annotations/images"
    labels_dir = "/home/makish/Desktop/Science_fair_2025/Annotations/labels"
    output_dir = "/home/makish/Desktop/Science_fair_2025/Annotations"

    split_dataset(images_dir, labels_dir, output_dir)
