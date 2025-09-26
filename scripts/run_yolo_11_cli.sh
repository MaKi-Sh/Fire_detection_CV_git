yolo detect predict model=yolo11n.pt source="gst-launch-1.0 nvarguscamerasrc ! nveglglessink"
#error code: A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/home/nvidia/.local/bin/yolo", line 3, in <module>
    from ultralytics.cfg import entrypoint
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/__init__.py", line 12, in <module>
    from ultralytics.utils import ASSETS, SETTINGS
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/utils/__init__.py", line 25, in <module>
    import torch
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/__init__.py", line 2422, in <module>
    from torch import (
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/export/__init__.py", line 64, in <module>
    from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/export/dynamic_shapes.py", line 18, in <module>
    from .exported_program import ExportedProgram
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/export/exported_program.py", line 24, in <module>
    from torch._higher_order_ops.utils import autograd_not_implemented
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/_higher_order_ops/__init__.py", line 1, in <module>
    from torch._higher_order_ops.cond import cond
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/_higher_order_ops/cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/_subclasses/functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/_subclasses/functional_tensor.py", line 267, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
/home/nvidia/.local/lib/python3.10/site-packages/torch/_subclasses/functional_tensor.py:267: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
Ultralytics 8.3.203 🚀 Python-3.10.12 torch-2.5.0a0+872d972e41.nv24.08 CUDA:0 (Orin, 7620MiB)
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
  File "/home/nvidia/.local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 36, in generator_context
    response = gen.send(None)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/engine/predictor.py", line 306, in stream_inference
    self.setup_source(source if source is not None else self.args.source)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/engine/predictor.py", line 261, in setup_source
    self.dataset = load_inference_source(
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/data/build.py", line 310, in load_inference_source
    dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)
  File "/home/nvidia/.local/lib/python3.10/site-packages/ultralytics/data/loaders.py", line 376, in __init__
    raise FileNotFoundError(f"{p} does not exist")
FileNotFoundError: gst-launch-1.0 nvarguscamerasrc ! nveglglessink does not exist
