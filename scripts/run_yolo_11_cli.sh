yolo detect predict model=yolo11n.pt \
  source="nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink" \
  show=True


WARNING ⚠️ torchvision==0.20 is incompatible with torch==2.8.
Run 'pip install torchvision==0.23' to fix torchvision or 'pip install -U torch torchvision' to update both.
For a full compatibility table see https://github.com/pytorch/vision#installation
Ultralytics 8.3.203 🚀 Python-3.10.12 torch-2.8.0+cpu CPU (ARMv8 Processor rev 1 (v8l))
YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs

Traceback (most recent call last):
  File "/home/nvidia/.local/bin/yolo", line 7, in <module>
    sys.exit(entrypoint())
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/cfg/__init__.py", line 990, in entrypoint
    getattr(model, mode)(**overrides)  # default args from model
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/engine/model.py", line 557, in predict
    return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/engine/predictor.py", line 249, in predict_cli
    for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 38, in generator_context
    response = gen.send(None)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/engine/predictor.py", line 306, in stream_inference
    self.setup_source(source if source is not None else self.args.source)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/engine/predictor.py", line 261, in setup_source
    self.dataset = load_inference_source(
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/data/build.py", line 310, in load_inference_source
    dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/data/loaders.py", line 376, in __init__
    raise FileNotFoundError(f"{p} does not exist")
FileNotFoundError: nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink does not exist
