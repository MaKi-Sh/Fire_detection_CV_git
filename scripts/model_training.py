from ultralytics import YOLO, settings
import os
import wandb

os.environ["WANDB_PROJECT"] = "Fire-detect_absparamwinner_1"
settings.update(wandb=True)

wandb.init(project="Fire_detect-Different_model")
wandb.define_metric("metrics/precision(B)", summary="max")
wandb.define_metric("metrics/recall(B)", summary="max")
wandb.define_metric("metrics/mAP50(B)", summary="max")

model = YOLO('yolo11s.pt')  # load model

model.train(data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config.yaml',
            name='small_model',
	    epochs=100,
            patience=20,
            imgsz=640,
            batch=16,
            device=0,
            project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs', 
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

wandb.finish()


















