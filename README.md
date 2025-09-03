---

language:

* en
  license: various
  tags:
* fire-detection
* computer-vision
* safety
* science-fair
  task\_categories:
* image-classification
* object-detection
* video-classification
  pretty\_name: Optical Fire Detection Dataset
  size\_categories:
* 10K\<n<100K

---

# 2025 Science Fair – Optical Fire Detection System

## Overview

This project explores an **optical fire detection system** using machine learning, designed to be faster and more robust than traditional smoke detectors.

### Limitations of traditional detectors

* **Photoelectric smoke alarms**: react slowly to fires.
* **Ionization detectors**: prone to false positives.
* **Aspirating Smoke Detection (ASD)**: can generate false positives, and in outdoor environments smoke sampling is difficult.

---

## Packages and Tools

### Data Preprocessing

* **Primary language**: Python
* **Libraries**: OpenCV (`opencv-python`)
* **Annotation**: CVAT

### Datasets

* **Fire**: FiresNet Dataset, BoWFire
* **Home + false positives**: Places365, Kinetics-700

### Training

* **Core libraries**

  * PyTorch (+ torchvision ops): baselines, losses, NMS, training loops
  * Lightning *or* Hydra: clean training loop & config management
  * Albumentations: train-time augmentations
  * Weights & Biases *or* TensorBoard: metrics, PR curves, experiment comparisons

* **Detector choices** (start with one, compare later)

  * Ultralytics YOLO (v8/v11 “nano/tiny”): quick start, TensorRT-friendly
  * RT-DETR (tiny): anchor-free, strong export story
  * MMDetection (OpenMMLab): very powerful, steeper learning curve

* **RGB–IR fusion**

  * Early fusion: 4-channel first convolution layer in PyTorch
  * Late/mid fusion: dual-stream backbones with simple shared heads

* **Evaluation & analysis**

  * FiftyOne: error analysis, hard negative mining
  * scikit-learn: precision, recall, F1 score at thresholds

* **Quantization & export**

  * ONNX Runtime exporter (from PyTorch)
  * TensorRT tooling (test exportability before deployment)
  * PyTorch FX quantization (`torch.ao.quantization`) for INT8

* **Other helpful tools**

  * SuperGradients (Deci)
  * Detectron2
  * timm (backbones)
  * ClearML (experiment tracking)

### Inference and Running

**Prototype (Python)**

* TensorRT Python API: load `.engine`, run inference
* OpenCV: camera capture, preprocessing, overlays
* GStreamer with OpenCV: stable camera IO (`nvarguscamerasrc`)
* pynvml: GPU monitoring
* uvloop / asyncio / multiprocessing: keep capture–infer–post separate

**Production (C++)**

* TensorRT C++ API: maximum performance
* GStreamer with `nvarguscamerasrc`, `nvvidconv`, `nvstreammux`: zero-copy NVMM frames
* OpenCV (CUDA build): preprocessing and visualization fallback
* NVIDIA DeepStream SDK: end-to-end pipeline

  * `nvv4l2camerasrc → nvstreammux → nvinfer (TRT) → nvosd → sink`
  * Temporal logic can run in a probe
* Logging: spdlog or glog
* Config: yaml-cpp (thresholds, N-of-M settings)
* Deployment: systemd for autostart and watchdog

---

## Computer Vision Pipeline

### Data Preprocessing

1. Collect data

   * Hard negatives: everyday scenes, normal houses
   * Controlled and uncontrolled fires
   * First collect video clips, then scatter into frames
2. Annotation: label fire regions or related features
3. Data cleaning: reduce reframes, remove redundancy and low-quality data
4. Data synchronization: align infrared and visual camera data
5. Data augmentation: simulate angles, glare, and other variations
6. Image resizing
7. Train–validation–test split
8. Dataset packaging
9. Preprocessing verification

### Training

Two approaches:

1. **Fine-tuning a general model**

   * Pros: higher performance potential
   * Cons: easier, less learning experience
2. **Training from scratch**

   * Pros: more educational
   * Cons: weaker performance expected

**Fine-tuning pipeline**

1. Sanity check dataset
2. Freeze splits
3. Choose fusion method and input size
4. Build training dataloader
5. Run full fine-tune
6. Validate during training
7. Select operating point
8. Add temporal logic
9. Stress test on hold-out set
10. Quantization & export
11. On-device verification
12. Versioning and publishing
13. Active-learning loop

### Running on Device

1. Deploy model to Jetson
2. Build TensorRT engine
3. Connect RGB and IR cameras
4. TensorRT inference loop:

   * Capture frame(s)
   * Preprocess (resize, normalize, optionally concat IR as 4th channel)
   * Inference (batch = 1)
   * Post-process (NMS, decode boxes)
   * Temporal logic (N-of-M voting, hysteresis, RGB–IR agreement if late fusion)
   * Draw overlays and log events
   * Output: screen preview or MQTT/HTTP event
5. DeepStream (optional alternative)
6. Apply thresholds & temporal logic directly on device
7. Measure performance

