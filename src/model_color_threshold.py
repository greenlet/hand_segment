import cv2
import numpy as np


class ColorThreshold:
  skin_color_lower_hls = np.array([170, 80, 5])
  skin_color_upper_hls = np.array([255, 200, 90])
  blur_size = (15, 15)

  def process(self, img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    range_mask = cv2.inRange(img_hls,
      ColorThreshold.skin_color_lower_hls, ColorThreshold.skin_color_upper_hls)
    # cv2.imshow('mask', range_mask)
    blurred = cv2.blur(range_mask, ColorThreshold.blur_size)
    # cv2.imshow('blurred', blurred)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_OTSU)
    cv2.imshow('Color Threshold', thresholded)
    return thresholded

