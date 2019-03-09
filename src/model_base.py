import os
import cv2
from src import utils


class ModelBase:
  def __init__(self, opt):
    self._base_opt = opt
    if self._base_opt.save:
      self._init_save()

  def _init_save(self):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_dir = utils.abs_path('..', 'model_rec')
    utils.make_dir(video_dir)
    if self._base_opt.video_file_name_base:
      video_file_name = '{}_{}'.format(self._base_opt.name, self._base_opt.video_file_name_base)
    else:
      dt_str = utils.cur_datetime()
      video_file_name = 'source_{}_{:.0f}x{:.0f}_{:.0f}.avi'.format(
        dt_str, self._base_opt.width, self._base_opt.height, self._base_opt.fps)
    video_file_path = os.path.join(video_dir, video_file_name)
    print('Write to file', video_file_path)
    self._writer = cv2.VideoWriter(video_file_path, fourcc, self._base_opt.fps,
      (self._base_opt.width, self._base_opt.height), False)

  def _post_process(self, img):
    if self._base_opt.save:
      self._writer.write(img)

  def release(self):
    if self._base_opt.save:
      self._writer.release()
      self._writer = None


