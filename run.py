import argparse
import os
from src import utils
from src.model_color_mask import ColorMask
from src.model_opencv_subtractor import OpenCVSubtractor
from src.model_hgr_net import HGRNet


def run(opt):
  model_opt = utils.AttrDict(save=opt.save)
  source = opt.source
  if source.isdigit():
    source = int(source)
  else:
    file_name_base = os.path.split(source)[-1]
    if file_name_base.startswith('source_') and len(file_name_base) > 7:
      file_name_base = file_name_base[7:]
      if len(file_name_base) > 0:
        model_opt.video_file_name_base = file_name_base
  capturer = utils.Capturer(source)

  if opt.save:
    model_opt.width = capturer.width
    model_opt.height = capturer.height
    model_opt.fps = int(capturer.fps)

  procs = []
  if opt.color_mask_light:
    procs.append(ColorMask(light=True, **model_opt))
  if opt.color_mask_dark:
    procs.append(ColorMask(light=False, **model_opt))
  if opt.gauss_mixture:
    procs.append(OpenCVSubtractor(OpenCVSubtractor.GAUSS_MIXTURE, **model_opt))
  if opt.knn:
    procs.append(OpenCVSubtractor(OpenCVSubtractor.KNN, **model_opt))
  if opt.hgr_net:
    procs.append(HGRNet(dense=False, **model_opt))
  if opt.hgr_net_dense:
    procs.append(HGRNet(dense=True, **model_opt))

  for img in capturer.frames():
    for proc in procs:
      proc.process(img)
  
  for proc in procs:
    proc.release()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Hand segmentation models demo')
  parser.add_argument('--color-mask-light', action='store_true',
    help='HLS color mask with low Hue values')
  parser.add_argument('--color-mask-dark', action='store_true',
    help='HLS color mask with high Hue values')
  parser.add_argument('--gauss-mixture', action='store_true',
    help='OpenCV Gaussian Mixture-base subtractor (BackgroundSubtractorMOG2)')
  parser.add_argument('--knn', action='store_true',
    help='OpenCV K-nearest neighbour subtractor (BackgroundSubtractorKNN)')
  parser.add_argument('--hgr-net', action='store_true',
    help='HGR-Net model (CNN with ASPP)')
  parser.add_argument('--hgr-net-dense', action='store_true',
    help='HGR-Net model (CNN with Dense ASPP)')
  parser.add_argument('--source', type=str, default='0',
    help='Path for input file. When set to number N webcam N is used: cv2.VideoCapture(N)')
  parser.add_argument('--save', action='store_true',
    help="Save each model's output to file")

  opt = parser.parse_args()
  print('Options', opt)
  print('Press Esc or q (Q) to exit')
  print('Press f (F) to save frame')
  print('Press r (R) to toggle input recording')
  run(opt)


