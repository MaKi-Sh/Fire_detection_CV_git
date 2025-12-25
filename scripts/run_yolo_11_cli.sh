sudo apt-get install v4l2loopback-dkms v4l2loopback-utils

sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="VirtualCam" exclusive_caps=1

gst-launch-1.0 nvarguscamerasrc ! \
  video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! \
  nvvidconv ! video/x-raw,format=YUY2 ! v4l2sink device=/dev/video2

yolo predict model=yolo11n.pt source=2
