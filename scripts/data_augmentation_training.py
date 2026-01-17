from ultralytics import YOLO, settings
import os
import random
import numpy as np
import pandas as pd
import wandb

# Define baseline (all augmentations off)
baseline = {
    "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "translate": 0.0,
    "scale": 0.0, "shear": 0.0, "perspective": 0.0, "hsv_h": 0.0,
    "hsv_s": 0.0, "hsv_v": 0.0, "mosaic": 0.0, "mixup": 0.0,
    "copy_paste": 0.0, "erasing": 0.0, "close_mosaic": 0, "bgr": 0.0
}

# Define test values for each parameter
param_values = {
    "fliplr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "flipud": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "degrees": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
    "translate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "scale": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "shear": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
    "perspective": [0.0001,0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001],
    "hsv_h": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "hsv_s": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "hsv_v": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "mosaic": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "mixup": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "copy_paste": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "erasing": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "bgr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# Generate experiments: baseline + one param varied at a time
experiments = [{"name": "baseline", **baseline}]

for param, values in param_values.items():
    for val in values:
        exp = baseline.copy()
        exp[param] = val
        exp["name"] = f"{param}_{val}"
        experiments.append(exp)

print(f"Total experiments: {len(experiments)}")



os.environ["WANDB_PROJECT"] = "Fire-detect-augmentation"
settings.update(wandb=False)  # Disable per-trial logging

# Training loop for all experiments
num_trials = 5
for i, exp in enumerate(experiments):
  # Collect max metrics across trials
  max_precisions, max_recalls, max_mAP50s, seeds_used = [], [], [], []

  for trial in range(1, num_trials + 1):
    print(f"\n{'='*60}")
    print(f"Running experiment {i+1}/{len(experiments)}: {exp['name']} (trial {trial}/{num_trials})")
    print(f"{'='*60}\n")

    # Generate random seed for this trial
    seed = random.randint(0, 2**32 - 1)
    seeds_used.append(seed)
    print(f"Using seed: {seed}")

    # Load fresh model for each experiment
    model = YOLO('yolo11n.pt')

    # Extract augmentation params (exclude 'name')
    aug_params = {k: v for k, v in exp.items() if k != "name"}

    run_dir = f"/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs/{exp['name']}_trial{trial}"

    model.train(
        seed=seed,
        data='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/scripts/config.yaml',
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        device=0,
        project='/home/makish/Desktop/Science_fair_2025/Fire_detection_CV_git/runs',
        name=f"{exp['name']}_trial{trial}",
        **aug_params
    )

    # Read results CSV to get max values across all epochs
    results_csv = pd.read_csv(f"{run_dir}/results.csv")
    results_csv.columns = results_csv.columns.str.strip()
    max_precisions.append(results_csv['metrics/precision(B)'].max())
    max_recalls.append(results_csv['metrics/recall(B)'].max())
    max_mAP50s.append(results_csv['metrics/mAP50(B)'].max())

  # Log averaged max results to wandb (one run per experiment)
  run = wandb.init(
      project="Fire-detect-augmentation",
      name=exp['name'],
      reinit=True
  )
  wandb.config.update({**aug_params, "seeds": seeds_used})
  wandb.log({
      "avg_max_precision": np.mean(max_precisions),
      "avg_max_recall": np.mean(max_recalls),
      "avg_max_mAP50": np.mean(max_mAP50s),
      "std_max_precision": np.std(max_precisions),
      "std_max_recall": np.std(max_recalls),
      "std_max_mAP50": np.std(max_mAP50s),
  })
  wandb.finish()

print("\nAll experiments completed!")

