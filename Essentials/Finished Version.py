import os    # used to execute bash commands to download and uncompress stuff
import cv2   # OpenCV image processing library
import glob

from PIL import Image

import numpy as np  # For math
import matplotlib.pyplot as plt  # For plotting stuff
from pynq_dpu import DpuOverlay   # Overlay for the FPGA
from pynq_peripherals import PmodGroveAdapter
from time import sleep
from kv260 import BaseOverlay
from pynq_peripherals import ArduinoSEEEDGroveAdapter, PmodGroveAdapter
base = BaseOverlay('base.bit')
port = PmodGroveAdapter(base.PMODA, G2='grove_pir')
# setup led bar on the board 
overlay = DpuOverlay("dpu.bit")
# setup led bar on the board 
adapter = PmodGroveAdapter(overlay.PMODA, G3='grove_ledbar')
ledbar = adapter.G3
m = port.G2
f = 1
sample_interval = 0.1 #seconds
number_of_samples = 1
while f == 1:
    print('{:.1f}: {}'.format(number_of_samples*sample_interval, "motion detected" if m.motion_detected() else "-"))
    if m.motion_detected:
        dave_detected = True
    sleep(sample_interval)
    number_of_samples = number_of_samples + 1
    level = 0
    brightness = 3
    blue_to_red = 1
    # set the level on the led bar
    while daveDetected == True:
        for i in range(1,11):
            level = i
            ledbar.set_level(int(level), brightness, blue_to_red)
            sleep(0.01)
        if level  != 0:
            level = 0
        sleep(0.01)
    level = 0
    ledbar.set_level(int(level), brightness, blue_to_red)
