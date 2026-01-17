import os
import shutil

labels_dir = "/home/makish/Desktop/Science_fair_2025/Annotations/labels"
raw_images_dir = "/media/makish/0051-D5A7/Raw_files"
output_images_dir = "/home/makish/Desktop/Science_fair_2025/Annotations/images"

os.makedirs(output_images_dir, exist_ok=True)

label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

copied = 0
missing = 0

for label_file in label_files:
    base_name = os.path.splitext(label_file)[0]
    
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_path = os.path.join(raw_images_dir, base_name + ext)
        if os.path.exists(image_path):
            shutil.copy(image_path, output_images_dir)
            copied += 1
            break
    else:
        print(f"Missing image for: {label_file}")
        missing += 1

print(f"\nDone! Copied {copied} images, {missing} missing")
