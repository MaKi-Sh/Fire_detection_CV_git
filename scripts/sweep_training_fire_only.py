from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect import DetectionTrainer
import os
import wandb
import random
import yaml
import cv2
from pathlib import Path

# Fire class index for 2-class dataset: ['ControlledFire', 'Fire']
# ControlledFire = 0, Fire = 1
FIRE_CLASS_INDEX = 1
CONFIG_PATH = '/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only.yaml'


def visualize_validation_predictions(model, save_dir, num_images=5):
    """
    Run inference on random validation images and save visualizations.

    Args:
        model: Trained YOLO model
        save_dir: Directory to save the output images
        num_images: Number of images to visualize
    """
    # Load config to get validation path
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Get validation images path
    data_path = Path(config.get('path', ''))
    val_path = data_path / config.get('val', 'valid/images')

    if not val_path.exists():
        print(f"Validation path not found: {val_path}")
        return

    # Get all image files from validation set
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    val_images = [f for f in val_path.iterdir() if f.suffix.lower() in image_extensions]

    if len(val_images) == 0:
        print("No validation images found")
        return

    # Select random images
    num_to_select = min(num_images, len(val_images))
    selected_images = random.sample(val_images, num_to_select)

    # Create output directory
    output_dir = Path(save_dir) / 'validation_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating predictions for {num_to_select} validation images...")

    # Run inference and save results
    wandb_images = []
    for img_path in selected_images:
        results = model.predict(source=str(img_path), save=False, verbose=False)

        # Get the plotted image with predictions
        for r in results:
            plotted_img = r.plot()  # Returns BGR numpy array with boxes drawn

            # Save to file
            output_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), plotted_img)
            print(f"  Saved: {output_path}")

            # Prepare for wandb logging (convert BGR to RGB)
            if wandb.run is not None:
                rgb_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
                wandb_images.append(wandb.Image(rgb_img, caption=img_path.name))

    # Log images to wandb
    if wandb.run is not None and wandb_images:
        wandb.log({"validation_predictions": wandb_images})
        print(f"  Logged {len(wandb_images)} images to wandb")

    print(f"Validation predictions saved to: {output_dir}")

sweep_config = {
    "name": "yolo11n recall optimization",
    "method": "bayes",
    "metric": {"name": "metrics/recall(B)", "goal": "maximize"},
    "parameters": {
        # Core training - HIGH IMPACT
        "optimizer": {"values": ["SGD", "AdamW"]},  # These two are the main ones
        "lr0": {"min": 0.001, "max": 0.01},         # Narrower range
        "batch": {"values": [4, 8, 16]},
        "imgsz": {"values": [512, 640]},

        # Loss weights - IMPORTANT for prioritizing fire class
        "cls": {"min": 0.5, "max": 2.0},

        # Key augmentations for fire detection
        "hsv_h": {"min": 0.0, "max": 0.05},   # Fire color matters, keep small
        "hsv_s": {"min": 0.0, "max": 0.7},
        "hsv_v": {"min": 0.0, "max": 0.5},
        "mosaic": {"min": 0.0, "max": 1.0},
        "scale": {"min": 0.0, "max": 0.5},
    }
}

# Config for training (uses test set for validation during sweep)
TRAIN_CONFIG_PATH = '/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only_train.yaml'
# Config for final validation (uses val set)
VAL_CONFIG_PATH = '/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only.yaml'


def create_train_config():
    """Create a training config that uses test set for validation."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Modify to use test set for validation during training
    config['val'] = 'images/test'

    with open(TRAIN_CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created training config with val=images/test at {TRAIN_CONFIG_PATH}")


def train():
    """Training function called by wandb agent for each sweep run."""
    wandb.init()
    config = wandb.config

    # Define metrics with focus on fire class recall
    wandb.define_metric("fire_recall", summary="max")
    wandb.define_metric("fire_precision", summary="max")
    wandb.define_metric("metrics/recall(B)", summary="max")
    wandb.define_metric("metrics/mAP50(B)", summary="max")

    model = YOLO('yolo11n.pt')

    # Custom callback to log fire-class-specific metrics after each validation
    def log_fire_metrics(validator):
        """Extract and log fire class metrics after validation."""
        if wandb.run is None:
            return
        try:
            # Get per-class metrics from validator results
            # validator.metrics contains per-class recall/precision
            if hasattr(validator, 'metrics') and validator.metrics is not None:
                # Get recall for fire class (index 1 in 2-class dataset)
                if hasattr(validator.metrics, 'class_result'):
                    # class_result returns (p, r, ap50, ap) for the specified class
                    fire_p, fire_r, fire_ap50, fire_ap = validator.metrics.class_result(FIRE_CLASS_INDEX)
                    wandb.log({
                        "fire_recall": fire_r,
                        "fire_precision": fire_p,
                        "fire_ap50": fire_ap50,
                        "fire_ap": fire_ap,
                    }, commit=False)  # Don't advance step - let Ultralytics control it
        except Exception as e:
            print(f"Warning: Could not log fire metrics: {e}")

    # Add the callback to the model
    model.add_callback("on_val_end", log_fire_metrics)

    # Train using config that validates against test set
    results = model.train(
        data=TRAIN_CONFIG_PATH,  # Uses test set for validation
        name=f'sweep_fire_only_{wandb.run.id}',
        epochs=100,
        patience=20,
        device=0,
        project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
        # Sweep hyperparameters
        optimizer=config.optimizer,
        lr0=config.lr0,
        batch=config.batch,
        imgsz=config.imgsz,
        cls=config.cls,
        # Data augmentation
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        mosaic=config.mosaic,
        scale=config.scale,
        amp=True,  # Mixed precision to reduce VRAM usage
    )

    # Log final metrics summary (check if run is still active since Ultralytics may have closed it)
    if results and wandb.run is not None:
        final_metrics = {}

        # Log overall metrics from results_dict
        if hasattr(results, 'results_dict'):
            final_metrics.update({
                "final/recall_max": results.results_dict.get('metrics/recall(B)', 0),
                "final/precision_max": results.results_dict.get('metrics/precision(B)', 0),
                "final/mAP50_max": results.results_dict.get('metrics/mAP50(B)', 0),
                "final/mAP50-95_max": results.results_dict.get('metrics/mAP50-95(B)', 0),
            })

        # Log fire-class-specific final metrics
        try:
            if hasattr(results, 'box') and hasattr(results.box, 'class_result'):
                fire_p, fire_r, fire_ap50, fire_ap = results.box.class_result(FIRE_CLASS_INDEX)
                final_metrics.update({
                    "final/fire_recall": fire_r,
                    "final/fire_precision": fire_p,
                    "final/fire_ap50": fire_ap50,
                    "final/fire_ap": fire_ap,
                })
        except Exception as e:
            print(f"Warning: Could not log final fire metrics: {e}")

        if final_metrics:
            wandb.log(final_metrics, commit=False)  # Don't advance step

    # Visualize predictions on 5 random validation images
    try:
        # Get the save directory from training results
        if hasattr(results, 'save_dir'):
            save_dir = results.save_dir
        else:
            save_dir = f'/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs/sweep_fire_only_{wandb.run.id if wandb.run else "unknown"}'

        # Load the best model weights for inference
        best_weights = Path(save_dir) / 'weights' / 'best.pt'
        if best_weights.exists():
            best_model = YOLO(str(best_weights))
            visualize_validation_predictions(best_model, save_dir, num_images=5)
        else:
            print(f"Best weights not found at {best_weights}, using current model")
            visualize_validation_predictions(model, save_dir, num_images=5)
    except Exception as e:
        print(f"Warning: Could not visualize validation predictions: {e}")

    # Final evaluation on val set (does NOT influence sweep scores)
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET (not influencing sweep)")
    print("="*60)
    try:
        best_weights = Path(save_dir) / 'weights' / 'best.pt'
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
        else:
            val_model = model

        # Evaluate on actual validation set using original config
        val_results = val_model.val(data=VAL_CONFIG_PATH, split='val')

        # Print val set metrics (not logged to wandb to avoid influencing sweep)
        print("\nValidation Set Results (for reference only):")
        if hasattr(val_results, 'results_dict'):
            print(f"  mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"  Precision: {val_results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"  Recall: {val_results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")

        # Log fire class metrics from val set
        if hasattr(val_results, 'box') and hasattr(val_results.box, 'class_result'):
            fire_p, fire_r, fire_ap50, fire_ap = val_results.box.class_result(FIRE_CLASS_INDEX)
            print(f"\n  Fire Class (val set):")
            print(f"    Recall: {fire_r:.4f}")
            print(f"    Precision: {fire_p:.4f}")
            print(f"    AP50: {fire_ap50:.4f}")
            print(f"    AP: {fire_ap:.4f}")

    except Exception as e:
        print(f"Warning: Could not run final validation: {e}")

    print("="*60 + "\n")

    # Only finish if run is still active
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    settings.update(wandb=True)

    # Create training config that uses test set for validation
    create_train_config()

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="Fire-detect_yolo11n_recall_opt")

    # Run the sweep agent (change count to set number of runs)
    wandb.agent(sweep_id, function=train, count=120)
