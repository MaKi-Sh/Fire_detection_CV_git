from ultralytics import YOLO, settings
import os
import wandb

os.environ["WANDB_PROJECT"] = "Fire-detect_confidence_test"
settings.update(wandb=True)

wandb.init(project="Fire-detect_confidence_test")
wandb.define_metric("metrics/precision(B)", summary="max")
wandb.define_metric("metrics/recall(B)", summary="max")
wandb.define_metric("metrics/mAP50(B)", summary="max")

# Train the model once
print("=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = YOLO('yolo11n.pt')

results = model.train(data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config.yaml',
            epochs=100,
            patience=20,
            imgsz=640,
            batch=16,
            device=0,
            project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
            name='confidence_test_run',
            #data augmentation
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.2,
            fliplr=0.1,
            bgr=0.0,
            mosaic=1.0,
            mixup=0.7,
            copy_paste=0.0,
            copy_paste_mode="flip",
            auto_augment=None,
            erasing=0.0,
            crop_fraction=0.0
)

# Get the path to the best trained model
best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
print(f"\nBest model saved at: {best_model_path}")

# Load the trained model for validation
trained_model = YOLO(best_model_path)

# Test with different confidence thresholds from 0.1 to 1.0
print("\n" + "=" * 60)
print("TESTING WITH DIFFERENT CONFIDENCE THRESHOLDS")
print("=" * 60)

confidence_results = []

for conf in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"\n{'='*60}")
    print(f"Testing with confidence threshold: {conf}")
    print(f"{'='*60}")

    val_results = trained_model.val(
        data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config.yaml',
        conf=conf,
        device=0
    )

    result_dict = {
        'confidence': conf,
        'mAP50': val_results.box.map50,
        'mAP50-95': val_results.box.map,
        'precision': val_results.box.mp,
        'recall': val_results.box.mr
    }
    confidence_results.append(result_dict)

    print(f"Confidence: {conf}")
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall: {val_results.box.mr:.4f}")

# Print summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Conf':<8} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 48)
for r in confidence_results:
    print(f"{r['confidence']:<8.1f} {r['mAP50']:<10.4f} {r['mAP50-95']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")

wandb.finish()







