from ultralytics import YOLO, settings
import os
import wandb




sweep_config = {
    "name": "yolo11s optimization",
    "method": "bayes",
    "metric": {"name": "metrics/recall(B).max", "goal": "maximize"},
    "parameters": {
        "optimizer": {"values": ["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]},
        "lr0": {"min": 0.00001, "max": 0.1},
        "lrf": {"min": 0.01, "max": 1.0},
        "momentum": {"min": 0.6, "max": 0.98},
        "weight_decay": {"min": 0.0, "max": 0.001},
        "warmup_epochs": {"min": 0.0, "max": 5.0},
        "batch": {"values": [8, 16, 32, 64, 128]},
        "imgsz": {"values": [416, 512, 640]},
        "box": {"min": 0.02, "max": 0.2},
        "cls": {"min": 0.2, "max": 4.0},
        "dfl": {"min": 0.5, "max": 2.0},
        "hsv_h": {"min": 0.0, "max": 0.1},
        "hsv_s": {"min": 0.0, "max": 0.9},
        "hsv_v": {"min": 0.0, "max": 0.9},
        "degrees": {"min": 0.0, "max": 45.0},
        "translate": {"min": 0.0, "max": 0.9},
        "scale": {"min": 0.0, "max": 0.9},
        "shear": {"min": 0.0, "max": 10.0},
        "perspective": {"min": 0.0, "max": 0.001},
        "flipud": {"min": 0.0, "max": 1.0},
        "fliplr": {"min": 0.0, "max": 1.0},
        "bgr": {"min": 0.0, "max": 1.0},
        "mosaic": {"min": 0.0, "max": 1.0},
        "mixup": {"min": 0.0, "max": 1.0},
        "copy_paste": {"min": 0.0, "max": 1.0},
        "erasing": {"min": 0.0, "max": 0.9},
        "crop_fraction": {"min": 0.1, "max": 1.0},
        "cos_lr": {"values": [True, False]},
        "close_mosaic": {"min": 0, "max": 20},
        "dropout": {"min": 0.0, "max": 0.5},
        "label_smoothing": {"min": 0.0, "max": 0.2},
        "nbs": {"values": [16, 32, 64]},
        "freeze": {"min": 0, "max": 10},
    }
}











def train():
    """Training function called by wandb agent for each sweep run."""
    wandb.init()
    config = wandb.config

    wandb.define_metric("metrics/precision(B)", summary="max")
    wandb.define_metric("metrics/recall(B)", summary="max")
    wandb.define_metric("metrics/mAP50(B)", summary="max")

    model = YOLO('yolo11s.pt')

    results = model.train(
        data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config.yaml',
        name=f'sweep_{wandb.run.id}',
        epochs=100,
        patience=20,
        device=0,
        project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
        # Sweep hyperparameters
        optimizer=config.optimizer,
        lr0=config.lr0,
        lrf=config.lrf,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        batch=config.batch,
        imgsz=config.imgsz,
        box=config.box,
        cls=config.cls,
        dfl=config.dfl,
        # Data augmentation
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        shear=config.shear,
        perspective=config.perspective,
        flipud=config.flipud,
        fliplr=config.fliplr,
        bgr=config.bgr,
        mosaic=config.mosaic,
        mixup=config.mixup,
        copy_paste=config.copy_paste,
        erasing=config.erasing,
        crop_fraction=config.crop_fraction,
        cos_lr=config.cos_lr,
        close_mosaic=config.close_mosaic,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
        nbs=config.nbs,
        freeze=config.freeze,
    )

    # Log final metrics summary (check if run is still active since Ultralytics may have closed it)
    if results and hasattr(results, 'results_dict') and wandb.run is not None:
        wandb.log({
            "final/recall_max": results.results_dict.get('metrics/recall(B)', 0),
            "final/precision_max": results.results_dict.get('metrics/precision(B)', 0),
            "final/mAP50_max": results.results_dict.get('metrics/mAP50(B)', 0),
            "final/mAP50-95_max": results.results_dict.get('metrics/mAP50-95(B)', 0),
        })

    # Only finish if run is still active
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    settings.update(wandb=True)

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="Fire-detect_yolo11s_optimization")

    # Run the sweep agent (change count to set number of runs)
    wandb.agent(sweep_id, function=train, count=330)





