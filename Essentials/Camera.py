#Main import block
import cv2 as camera
import numpy as np
import matplotlib.pyplot as plt
import os

# Import overlay
from pynq_dpu import DPUOverlay
overlay = DpuOverlay("base.bit")

# Load the model for AI
overlay.load_model("tf_yolov3_voc.xmodel")

# Video Code
image_outputs = []
videoIn = camera.VideoCapture(0)

