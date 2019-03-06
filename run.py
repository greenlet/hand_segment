import argparse
import cv2
from src import utils
from src.model_color_threshold import ColorThreshold
from src.model_opencv_subtractor import OpenCVSubtractor
from src.model_hgr_net import HGRNet


def run(opt):
  models = []
  if opt.color_threshold:
    models.append(ColorThreshold())
  if opt.gauss_mixture:
    models.append(OpenCVSubtractor(OpenCVSubtractor.GAUSS_MIXTURE))
  if opt.knn:
    models.append(OpenCVSubtractor(OpenCVSubtractor.KNN))
  if opt.hgr_net:
    models.append(HGRNet())
  if opt.hgr_net_dense:
    models.append(HGRNet(dense=True))
  
  if len(models) == 0:
    models.append(ColorThreshold())

  for img in utils.capture(0):
    cv2.imshow('Source', img)
    for model in models:
      model.process(img)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Hand segmentation models demo')
  parser.add_argument('--color-threshold', action='store_true',
    help='HLS color mask')
  parser.add_argument('--gauss-mixture', action='store_true',
    help='OpenCV Gaussian Mixture-base subtractor (BackgroundSubtractorMOG2)')
  parser.add_argument('--knn', action='store_true',
    help='OpenCV K-nearest neighbour subtractor (BackgroundSubtractorKNN)')
  parser.add_argument('--hgr-net', action='store_true',
    help='HGR-Net model (CNN with ASPP)')
  parser.add_argument('--hgr-net-dense', action='store_true',
    help='HGR-Net model (CNN with Dense ASPP)')
  opt = parser.parse_args()
  print('Options', opt)
  run(opt)


