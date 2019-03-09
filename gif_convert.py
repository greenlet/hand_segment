import os
import argparse
import subprocess
from src import utils

def run(opt):
  path_pairs = []
  src_path = os.path.abspath(opt.src)
  dst_path = os.path.abspath(opt.dst)
  utils.make_dir(dst_path)
  if os.path.isfile(src_path):
    dir_path, file_name = os.path.split(src_path)
    src_files = [(os.path.join(dir_path, file_name), file_name)]
  else:
    src_files = utils.list_files(src_path)
  for src_file_path, src_file_name in src_files:
    dst_file_name = os.path.splitext(src_file_name)[0] + '.gif'
    dst_file_path = os.path.join(dst_path, dst_file_name)
    path_pairs.append((src_file_path, dst_file_path))
  
  for fpath_src, fpath_dst in path_pairs:
    cmd = 'ffmpeg -y -i {} {}'.format(fpath_src, fpath_dst)
    print('Exec:', cmd)
    subprocess.run(cmd)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Video to GIF converter')
  parser.add_argument('--src', type=str, required=True,
    help='Source video file or directory path')
  parser.add_argument('--dst', type=str, required=True,
    help='Output directory')

  opt = parser.parse_args()
  print('Options', opt)
  run(opt)


