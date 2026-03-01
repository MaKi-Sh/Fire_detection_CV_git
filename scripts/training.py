from ultralytics import YOLO, settings
import os
import wandb
import random
import glob

os.environ["WANDB_PROJECT"] = "Fire-detect_yolo11n"
settings.update(wandb=True)

wandb.init(project="Fire-detect_yolo11n")
wandb.define_metric("metrics/precision(B)", summary="max")
wandb.define_metric("metrics/recall(B)", summary="max")
wandb.define_metric("metrics/mAP50(B)", summary="max")
wandb.define_metric("metrics/fire_precision", summary="max")
wandb.define_metric("metrics/fire_recall", summary="max")
wandb.define_metric("metrics/fire_AP50", summary="max")

model = YOLO('yolo11n.pt')  # load model

results = model.train(data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only.yaml',
            epochs=100,
	    patience=20,
            imgsz=640,
            batch=16,
            device=0,
            project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
)

# Validate best model and log fire-class metrics
best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
best_model = YOLO(best_model_path)
val_metrics = best_model.val(data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only.yaml', device=0)

FIRE_CLASS_IDX = 1
# ap_class_index maps position in the metrics arrays to actual class ids
class_indices = val_metrics.box.ap_class_index.tolist()
if FIRE_CLASS_IDX in class_indices:
    pos = class_indices.index(FIRE_CLASS_IDX)
    fire_precision = float(val_metrics.box.p[pos])
    fire_recall = float(val_metrics.box.r[pos])
    fire_ap50 = float(val_metrics.box.ap50[pos])
    wandb.log({
        "metrics/fire_precision": fire_precision,
        "metrics/fire_recall": fire_recall,
        "metrics/fire_AP50": fire_ap50,
    })
    print(f"\nFire class metrics - Precision: {fire_precision:.4f}, Recall: {fire_recall:.4f}, AP50: {fire_ap50:.4f}")
else:
    print("\nWarning: Fire class not found in validation results")

# Visualize predictions on 5 validation images
val_images_path = '/home/makish/Desktop/Science_fair_2025/Annotations/More_annotations/images/val'
val_images = glob.glob(os.path.join(val_images_path, '*.jpg')) + \
             glob.glob(os.path.join(val_images_path, '*.png')) + \
             glob.glob(os.path.join(val_images_path, '*.jpeg'))

# Select 5 random validation images
sample_images = random.sample(val_images, min(5, len(val_images)))

# Create output directory for visualizations
output_dir = os.path.join(results.save_dir, 'val_predictions')
os.makedirs(output_dir, exist_ok=True)

# Run inference and save images with bounding boxes
print(f"\nSaving predictions on {len(sample_images)} validation images to {output_dir}")
for img_path in sample_images:
    pred_results = best_model.predict(img_path, save=True, project=output_dir, name='', exist_ok=True)
    print(f"  Processed: {os.path.basename(img_path)}")

print(f"\nValidation predictions saved to: {output_dir}")
