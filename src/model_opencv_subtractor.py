import cv2
import numpy as np


class OpenCVSubtractor:
  GAUSS_MIXTURE = 'Gaussian Mixture'
  KNN = 'K-nearest neighbours'
  blur_size = (15, 15)

  def __init__(self, sub_type):
    self._sub_type = sub_type
    if sub_type == OpenCVSubtractor.GAUSS_MIXTURE:
      self._bg_sub = cv2.createBackgroundSubtractorMOG2()
    elif sub_type == OpenCVSubtractor.KNN:
      self._bg_sub = cv2.createBackgroundSubtractorKNN()
  
  def process(self, img):
    img_mask = self._bg_sub.apply(img)
    img_blur = cv2.GaussianBlur(img_mask, OpenCVSubtractor.blur_size, 0)
    _, img_res = cv2.threshold(img_blur, 127, 255, cv2.THRESH_OTSU)
    h, w = img_mask.shape
    shape = (h, 2 * w)
    img_out = np.zeros(shape, np.uint8)
    img_out[:, :w] = img_mask
    img_out[:, w:] = img_res
    cv2.imshow(self._sub_type, img_out)
    return img_res

