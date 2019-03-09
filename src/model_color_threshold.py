import cv2
import numpy as np
from src import utils
from src.model_base import ModelBase


class ColorThreshold(ModelBase):
  hue = (0, 30)
  saturation = (0, 150)
  lightness = (40, 150)
  lower_hls = np.array([hue[0], lightness[0], saturation[0]])
  upper_hls = np.array([hue[1], lightness[1], saturation[1]])
  blur_size = (15, 15)

  def __init__(self, **opt):
    base_opt = utils.AttrDict(name='color_threshold', **opt)
    if base_opt.width:
      base_opt.width *= 2
    super().__init__(base_opt)
    print('HSL: {} <= H <= {}, {} <= S <= {}, {} <= L <= {}'.format(
      *self.hue, *self.saturation, *self.lightness))

  def process(self, img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_mask = cv2.inRange(img_hls, self.lower_hls, self.upper_hls)
    blurred = cv2.blur(img_mask, self.blur_size)
    # cv2.imshow('blurred', blurred)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_OTSU)
    h, w = thresholded.shape
    shape = (h, 2 * w)
    img_out = np.zeros(shape, np.uint8)
    img_out[:, :w] = img_mask
    img_out[:, w:] = thresholded
    cv2.imshow('Color Threshold', img_out)
    super()._post_process(img_out)
    return thresholded

