import cv2
import time
from datetime import datetime
import os
import math
from PIL import Image
import numpy as np
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FPS_Counter:
    def __init__(self, name=None, lapse_sec=5):
        self._tag = '{} '.format(name) if name else ''
        self._lapse_sec = lapse_sec
        self._start = time.time()
        self._frames = 0

    def next(self):
        self._frames += 1
        now = time.time()
        delta_sec = now - self._start
        if delta_sec >= self._lapse_sec:
            fps = self._frames / delta_sec
            self._start = now
            # print('--> {:0.2} - {}'.format(delta_sec, self._frames))
            self._frames = 0
            print('{}FPS:{:.1f}'.format(self._tag, fps))


def cur_datetime():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_frame(img):
    width, height = img.shape[1], img.shape[0]
    dt_str = cur_datetime()
    file_name = 'frame_{}_{:.0f}x{:.0f}.jpg'.format(dt_str, width, height)
    print('Saving frame: {}'.format(file_name))
    if __file__ in dir():
        path = os.path.split(__file__)[0]
    else:
        path = ''
    path = abs_path(path, '..', 'screens')
    make_dir(path)
    file_path = os.path.join(path, file_name)
    print(file_path)
    cv2.imwrite(file_path, img)


def capture(cam_num=0):
    print('OpenCV {}'.format(cv2.__version__))
    cap = cv2.VideoCapture(cam_num)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('OpenCV capture resolution: {:.0f}x{:.0f} {}'.format(
        width, height, fps))

    fpsc = FPS_Counter()
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        yield img
        fpsc.next()

        key = cv2.waitKey(1)
        if key >= 0:
            print('key pressed: {}'.format(key))
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
            elif key == 49:
                save_frame(img)

    cap.release()


def make_dir(*subpaths):
    path = abs_path(*subpaths)
    os.makedirs(path, exist_ok=True)
    return path


def abs_path(*subpaths):
    path = os.path.join(*subpaths)
    if os.path.isabs(path):
        return path
    if '__file__' in globals():
        parts = os.path.split(__file__)[:-1] + subpaths
        return os.path.join(*parts)
    return os.path.abspath(*subpaths)


def list_files(*subpaths, paths_only=False):
    dir_path = abs_path(*subpaths)
    res = []
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        if os.path.isfile(path):
            if (paths_only):
                res.append(path)
            else:
                res.append((path, f))
    return res


def clear_dir(*subpaths):
    dir_path = abs_path(*subpaths)
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def clear_or_make_dir(*subpaths):
    dir_path = abs_path(*subpaths)
    if os.path.exists(dir_path):
        clear_dir(dir_path)
    else:
        make_dir(dir_path)


def copy_files(from_path, to_path, clear_first=True, move=False, rename_cb=None):
    from_path = abs_path(from_path) if type(from_path) == str else abs_path(*from_path)
    to_path = abs_path(to_path) if type(to_path) == str else abs_path(*to_path)
    if os.path.exists(to_path):
        if clear_first:
            clear_dir(to_path)
    else:
        make_dir(to_path)
    for from_file_path, from_file_name in list_files(from_path, with_names=True):
        to_file_name = rename_cb(from_file_name) if rename_cb else from_file_name
        to_file_path = os.path.join(to_path, to_file_name)
        shutil.copy(from_file_path, to_file_path)


def get_square(image, out_size=None):
    w, h = image.size
    sz = min(w, h)
    if not out_size or out_size > sz:
        out_size = sz
    x_offset = (w - sz) // 2
    y_offset = (h - sz) // 2
    image = image.crop((x_offset, y_offset, x_offset + sz, y_offset + sz))
    image.thumbnail((out_size, out_size))
    return image


def fit_image(images, size, mask_rect=None, crop=False):
    if type(images) != tuple and type(images) != list:
        ret_as_list = False
        images = [images]
    else:
        ret_as_list = True
        images = images
    if type(size) == int:
        w_dst, h_dst = size, size
    else:
        w_dst, h_dst = size
    w_src, h_src = images[0].size

    w_dst_h_src = w_dst * h_src
    w_src_h_dst = w_src * h_dst

    if w_dst_h_src == w_src_h_dst:
        if w_dst != w_src:
            for i in range(len(images)):
                images[i] = images[i].resize((w_dst, h_dst), Image.BILINEAR)
        return images if ret_as_list else images[0]

    r_dst = w_dst / h_dst
    if w_dst_h_src > w_src_h_dst:
        w_dst_fit = w_src
        h_dst_fit = math.ceil(w_src / r_dst)
    else:
        w_dst_fit = math.ceil(h_src * r_dst)
        h_dst_fit = h_src

    if mask_rect:
        x1, y1, x2, y2 = mask_rect
        w, h = x2 - x1, y2 - y1
        if w_dst_fit < w or h_dst_fit < h:
            if not crop:
                return None
            if w_dst_fit < w:
                x1 += (w - w_dst_fit) // 2
                x2 = x1 + w_dst_fit
                w = w_dst_fit
            if h_dst_fit < h:
                y1 += (h - h_dst_fit) // 2
                y2 = y1 + h_dst_fit
                h = h_dst_fit
        x1_crop = max(x1 - (w_dst_fit - w) // 2, 0)
        y1_crop = max(y1 - (h_dst_fit - h) // 2, 0)
    else:
        x1_crop = (w_src - w_dst_fit) // 2
        y1_crop = (h_src - h_dst_fit) // 2
    x2_crop = x1_crop + w_dst_fit
    y2_crop = y1_crop + h_dst_fit
    rect_crop = x1_crop, y1_crop, x2_crop, y2_crop

    for i in range(len(images)):
        images[i] = images[i].crop(rect_crop)
        if w_dst != w_dst_fit:
            images[i] = images[i].resize((w_dst, h_dst), Image.BILINEAR)

    return images if ret_as_list else images[0]


def tile_images(images, size=None, margin=None):
    n = len(images)
    if not size:
        w, h = images[0].size
        if not margin:
            margin = max(w // 10, 10)
        n_hor = math.ceil(math.sqrt(n * w / h))
        n_ver = math.ceil(n / n_hor)
        size = ((w + margin) * n_hor + margin, (h + margin) * n_ver + margin)
        if (size[0] > 1600 or size[1] > 1200):
            size = (1600, 1200)
    width, height = size
    n_hor = math.ceil(math.sqrt(n * width / height))
    n_ver = math.ceil(n / n_hor)
    wc = (width - margin) // n_hor
    hc = (height - margin) // n_ver
    w = wc - margin
    h = hc - margin
    res = Image.new('RGB', (width, height), color='white')
    x_ind, x_offset, y_offset = 0, margin, margin
    for i, image in enumerate(images, 1):
        image = image.resize((w, h), Image.BILINEAR)
        res.paste(image, (x_offset, y_offset))
        x_ind = (x_ind + 1) % n_hor
        if x_ind == 0:
            x_offset = margin
            y_offset += hc
        else:
            x_offset += wc
        x_offset = math.ceil(x_offset)
        y_offset = math.ceil(y_offset)

    return res


def img_white_balance(img, white_ratio):
    for channel in range(img.shape[2]):
        channel_max = np.percentile(img[:, :, channel], 100-white_ratio)
        channel_min = np.percentile(img[:, :, channel], white_ratio)
        img[:, :, channel] = (channel_max - channel_min) * (img[:, :, channel] / 255.0)
    return img


def find_mask_rect(mask_img):
    h, w = mask_img.shape
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for x in range(w):
        for y in range(h):
            if mask_img[y, x]:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
    if x_min > x_max:
        return None
    return (x_min, y_min, x_max, y_max)


