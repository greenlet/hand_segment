import cv2
import numpy as np
from src import utils
from src.model_base import ModelBase


class ColorMask(ModelBase):
  hue_light_range = (0, 30)
  hue_dark_range = (140, 180)
  saturation_range = (0, 150)
  lightness_range = (40, 150)
  blur_size = (15, 15)

  def __init__(self, light=True, **opt):
    if light:
      suffix = '_light'
      hue_range = self.hue_light_range
      self._name = 'Color Light Mask'
    else:
      suffix = '_dark'
      hue_range = self.hue_dark_range
      self._name = 'Color Dark Mask'
    base_name = name='color_mask' + suffix
    base_opt = utils.AttrDict(name=base_name, **opt)
    if base_opt.width:
      base_opt.width *= 2
    super().__init__(base_opt)
    self._lower_hls = np.array([hue_range[0], self.lightness_range[0], self.saturation_range[0]])
    self._upper_hls = np.array([hue_range[1], self.lightness_range[1], self.saturation_range[1]])
    print('HSL: {} <= H <= {}, {} <= S <= {}, {} <= L <= {}'.format(
      *hue_range, *self.saturation_range, *self.lightness_range))

  def process(self, img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = img_hls[:, :, 0], img_hls[:, :, 1], img_hls[:, :, 2]
    # print('h', h.min(), h.max())
    # print('l', l.min(), l.max())
    # print('s', s.min(), s.max())
    img_mask = cv2.inRange(img_hls, self._lower_hls, self._upper_hls)
    img_blur = cv2.GaussianBlur(img_mask, self.blur_size, 0)
    # cv2.imshow('blurred', img_blur)
    _, img_res = cv2.threshold(img_blur, 127, 255, cv2.THRESH_OTSU)
    h, w = img_res.shape
    shape = (h, 2 * w)
    img_out = np.zeros(shape, np.uint8)
    img_out[:, :w] = img_mask
    img_out[:, w:] = img_res
    cv2.imshow(self._name, img_out)
    super()._post_process(img_out)
    return img_res

