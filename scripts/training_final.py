from ultralytics import YOLO, settings
import shutil
import wandb
import random
import yaml
import cv2
from pathlib import Path

BEST_MODELS_DIR = Path('/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs/Best_models')

# Fire class index for 2-class dataset: ['ControlledFire', 'Fire']
FIRE_CLASS_INDEX = 1
CONFIG_PATH = '/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only.yaml'

# Uses test set for validation during training, val set for final eval
TRAIN_CONFIG_PATH = '/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only_train.yaml'
VAL_CONFIG_PATH = '/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config_fire_only.yaml'

MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
}


def create_train_config():
    """Create a training config that uses test set for validation."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['val'] = 'images/test'

    with open(TRAIN_CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created training config with val=images/test at {TRAIN_CONFIG_PATH}")


def visualize_validation_predictions(model, save_dir, num_images=5):
    """Run inference on random validation images and save visualizations."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    data_path = Path(config.get('path', ''))
    val_path = data_path / config.get('val', 'valid/images')

    if not val_path.exists():
        print(f"Validation path not found: {val_path}")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    val_images = [f for f in val_path.iterdir() if f.suffix.lower() in image_extensions]

    if len(val_images) == 0:
        print("No validation images found")
        return

    num_to_select = min(num_images, len(val_images))
    selected_images = random.sample(val_images, num_to_select)

    output_dir = Path(save_dir) / 'validation_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating predictions for {num_to_select} validation images...")

    wandb_images = []
    for img_path in selected_images:
        results = model.predict(source=str(img_path), save=False, verbose=False)

        for r in results:
            plotted_img = r.plot()
            output_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), plotted_img)
            print(f"  Saved: {output_path}")

            if wandb.run is not None:
                rgb_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
                wandb_images.append(wandb.Image(rgb_img, caption=img_path.name))

    if wandb.run is not None and wandb_images:
        wandb.log({"validation_predictions": wandb_images})
        print(f"  Logged {len(wandb_images)} images to wandb")

    print(f"Validation predictions saved to: {output_dir}")


def train_baseline(model_key, weights_path):
    """Train a single model with default settings as a baseline."""
    project_name = f"Fire-detect_{model_key}_baseline"

    run = wandb.init(
        project=project_name,
        name=f"baseline_{model_key}",
        config={
            "model": model_key,
            "epochs": 100,
            "patience": 20,
            "imgsz": 640,
            "batch": 16,
        },
    )

    wandb.define_metric("fire_recall", summary="max")
    wandb.define_metric("fire_precision", summary="max")
    wandb.define_metric("metrics/recall(B)", summary="max")
    wandb.define_metric("metrics/mAP50(B)", summary="max")

    model = YOLO(weights_path)

    def log_fire_metrics(validator):
        if wandb.run is None:
            return
        try:
            if hasattr(validator, 'metrics') and validator.metrics is not None:
                if hasattr(validator.metrics, 'class_result'):
                    fire_p, fire_r, fire_ap50, fire_ap = validator.metrics.class_result(FIRE_CLASS_INDEX)
                    wandb.log({
                        "fire_recall": fire_r,
                        "fire_precision": fire_p,
                        "fire_ap50": fire_ap50,
                        "fire_ap": fire_ap,
                    }, commit=False)
        except Exception as e:
            print(f"Warning: Could not log fire metrics: {e}")

    model.add_callback("on_val_end", log_fire_metrics)

    results = model.train(
        data=TRAIN_CONFIG_PATH,
        name=f'baseline_{model_key}',
        epochs=100,
        patience=20,
        device=0,
        project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
        imgsz=640,
        batch=16,
        amp=True,
    )

    # Log final metrics
    if results and wandb.run is not None:
        final_metrics = {}

        if hasattr(results, 'results_dict'):
            final_metrics.update({
                "final/recall_max": results.results_dict.get('metrics/recall(B)', 0),
                "final/precision_max": results.results_dict.get('metrics/precision(B)', 0),
                "final/mAP50_max": results.results_dict.get('metrics/mAP50(B)', 0),
                "final/mAP50-95_max": results.results_dict.get('metrics/mAP50-95(B)', 0),
            })

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
            wandb.log(final_metrics, commit=False)

    # Visualize predictions
    try:
        if hasattr(results, 'save_dir'):
            save_dir = results.save_dir
        else:
            save_dir = f'/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs/baseline_{model_key}'

        best_weights = Path(save_dir) / 'weights' / 'best.pt'
        if best_weights.exists():
            best_model = YOLO(str(best_weights))
            visualize_validation_predictions(best_model, save_dir, num_images=5)
        else:
            print(f"Best weights not found at {best_weights}, using current model")
            visualize_validation_predictions(model, save_dir, num_images=5)
    except Exception as e:
        print(f"Warning: Could not visualize validation predictions: {e}")

    # Final evaluation on val set
    print("\n" + "="*60)
    print(f"FINAL EVALUATION ON VALIDATION SET - {model_key}")
    print("="*60)
    try:
        best_weights = Path(save_dir) / 'weights' / 'best.pt'
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
        else:
            val_model = model

        val_results = val_model.val(data=VAL_CONFIG_PATH, split='val')

        print("\nValidation Set Results:")
        if hasattr(val_results, 'results_dict'):
            print(f"  mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"  Precision: {val_results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"  Recall: {val_results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")

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

    # Copy run folder to Best_models
    try:
        src = Path(save_dir)
        dst = BEST_MODELS_DIR / f"{project_name}_best"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied baseline run to: {dst}")
    except Exception as e:
        print(f"Warning: Could not copy to Best_models: {e}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    settings.update(wandb=True)

    create_train_config()
    BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_key, weights_path in MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Training baseline for {model_key}")
        print(f"{'#'*60}\n")

        train_baseline(model_key, weights_path)
