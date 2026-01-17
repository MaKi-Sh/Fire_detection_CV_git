from ultralytics import YOLO, settings
import os
import wandb

os.environ["WANDB_PROJECT"] = "Winners_fire_detect"
settings.update(wandb=True)

# Base config with all augmentations disabled
base_augment = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "scale": 0.0,
    "translate": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "erasing": 0.0,
}

experiments = [
    {"name": "winners", "mosaic": 1.0, "mixup": 0.7},
    {"name": "winners+safe", "mosaic": 1.0, "mixup": 0.7, "fliplr": 0.1, "flipud": 0.2, "scale": 0.3},
    {"name": "everything_helpful", "mosaic": 1.0, "mixup": 0.7, "fliplr": 0.1, "flipud": 0.2, "scale": 0.3, "hsv_v": 0.4, "perspective": 0.0002},
]

for exp in experiments:
    wandb.init(project="Winners_fire_detect", name=exp["name"], reinit=True)
    wandb.define_metric("metrics/precision(B)", summary="max")
    wandb.define_metric("metrics/recall(B)", summary="max")
    wandb.define_metric("metrics/mAP50(B)", summary="max")

    model = YOLO('yolo11n.pt')

    # Start with all zeros, then override with experiment values
    train_params = base_augment.copy()
    train_params.update({k: v for k, v in exp.items() if k != "name"})

    model.train(
        data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config.yaml',
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        device=0,
        project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
        name=exp["name"],
        **train_params,
    )

    wandb.finish()



