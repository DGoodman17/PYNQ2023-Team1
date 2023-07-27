#Main import block
import cv2 as camera
import numpy as np
import matplotlib.pyplot as plt
import os

# Import overlay
from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")

# Load the model for AI
overlay.load_model("tf_yolov3_voc.xmodel")

# Video Code
image_outputs = []
videoIn = camera.VideoCapture(0)
videoIn.set(camera.CAP_PROP_FRAME_WIDTH, 640)
videoIn.set(camera.CAP_PROP_FRAME_HEIGHT, 480)

# Read a frame and extract from camera
ret, frame = videoIn.read()

print("Program working")
del overlay