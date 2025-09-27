yolo detect predict model=yolo11n.pt \
  source="nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink" \
  show=True
