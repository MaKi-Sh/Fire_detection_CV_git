#load dataset 
dataset = dataset()
#Splitting dataset
from sklearn.model_selection import train_test_split
import os
import shutil
from ultralytics import YOLO

def split_dataset(images_dir, labels_dir, output_dir, test_size=0.2, val_size=0.2):
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=val_size, random_state=42)

    for subset, subset_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        os.makedirs(f"{output_dir}/images/{subset}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{subset}", exist_ok=True)
        for image in subset_images:
            shutil.copy(f"{images_dir}/{image}", f"{output_dir}/images/{subset}/{image}")
            label_file = image.replace('.jpg', '.txt')
            shutil.copy(f"{labels_dir}/{label_file}", f"{output_dir}/labels/{subset}/{label_file}")


model = YOLO(yolo11n.pt) #load model

model.train(data='config.yaml',  # Path to YAML config
            epochs=50,                  # Number of epochs
            imgsz=640,                  # Image size
            batch=16,                   # Batch size
            device=0)                   # GPU device index
#maybe configure weights and biases later 