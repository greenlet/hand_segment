import argparse
from src import utils
from src.model_color_threshold import ColorThreshold
from src.model_opencv_subtractor import OpenCVSubtractor
from src.model_hgr_net import HGRNet


def run(opt):
  procs = []
  if opt.color_threshold:
    procs.append(ColorThreshold())
  if opt.gauss_mixture:
    procs.append(OpenCVSubtractor(OpenCVSubtractor.GAUSS_MIXTURE))
  if opt.knn:
    procs.append(OpenCVSubtractor(OpenCVSubtractor.KNN))
  if opt.hgr_net:
    procs.append(HGRNet())
  if opt.hgr_net_dense:
    procs.append(HGRNet(dense=True))

  capturer = utils.Capturer()
  for img in capturer.frames(0):
    for model in procs:
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
  print('Press Esc or q (Q) to exit')
  print('Press f (F) to save frame')
  print('Press r (R) to toggle recording')
  run(opt)


