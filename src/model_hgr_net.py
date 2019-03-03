from keras import layers, models
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from src.bilinear import BilinearUpSampling2D
from src import utils


def img_get_square(img):
    h, w = img.shape[:2]
    sz = min(h, w)
    h_offset = (h - sz) // 2
    w_offset = (w - sz) // 2
    return img[h_offset:(h - h_offset), w_offset:(w - w_offset), :]


class HGRNet:
  def __init__(self):
    self._opt = utils.AttrDict(size=(320, 320), shape=(320, 320, 3))
    model_params_path = utils.abs_path('..', 'model_params', 'hgr_seg.hdf5')
    self._build_model()
    self._model.load_weights(model_params_path)

  def _relu(self, l):
    return layers.Activation('relu')(l)

  def _bn_relu(self, l):
    l = layers.BatchNormalization()(l)
    l = self._relu(l)
    return l

  def _conv_layer(self, l, filters, kernel_size, batch_normalization=False, dilation_rate=1, strides=1):
    l = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(l)
    if batch_normalization:
      l = self._bn_relu(l)
    else:
      l = self._relu(l)
    return l

  def _ASPP(self, l):
    dconv_filters = 32
    a1 = layers.Conv2D(dconv_filters, 1, activation='relu',
                padding='same', dilation_rate=1)(l)
    a2 = layers.Conv2D(dconv_filters, 3, activation='relu',
                padding='same', dilation_rate=3)(l)
    a3 = layers.Conv2D(dconv_filters, 3, activation='relu',
                padding='same', dilation_rate=6)(l)
    a4 = layers.Conv2D(dconv_filters, 3, activation='relu',
                padding='same', dilation_rate=10)(l)
    a5 = layers.Conv2D(dconv_filters, 3, activation='relu',
                padding='same', dilation_rate=15)(l)
    return layers.Concatenate(axis=-1)([a1, a2, a3, a4, a5])

  def _dense_ASPP(self, l, dilations=[3, 6, 12, 18, 24], dil_kernel_size=3):
    classes = l.shape[-1].value
    dil_filters_in = classes // 2
    dil_filters_out = classes // 4
    layers = [l]
    for dilation in dilations:
      if len(layers) == 1:
        l_dil = layers[0]
      else:
        l_dil = layers.Concatenate()(layers)
      l_dil = self._conv_layer(l_dil, dil_filters_in, 1, batch_normalization=True)
      l_dil = self._conv_layer(l_dil, dil_filters_out, dil_kernel_size, dilation_rate=dilation)
      layers.append(l_dil)
    return layers.Concatenate()(layers)

  def _res_net(self, l, filters, strides=1):
    filters_first = filters // 4
    l = self._bn_relu(l)
    l_prev = l
    l = self._conv_layer(l, filters_first, kernel_size=1, strides=strides, batch_normalization=True)
    l = self._conv_layer(l, filters_first, kernel_size=3, batch_normalization=True)
    l = layers.Conv2D(filters, kernel_size=1)(l)
    l_prev = layers.Conv2D(filters, kernel_size=1, strides=strides)(l_prev)
    l = layers.Add()([l_prev, l])
    return l

  def _res_group(self, l, filters, layers, strides):
    for i in range(layers):
      st = strides if i == 0 else 1
      l = self._res_net(l, filters, st)
    return l

  def _build_model(self):
    inp = layers.Input(self._opt.shape)
    l = inp
    
    l = layers.Conv2D(16, 7, padding='same')(l)
    l = self._res_group(l, 32, 3, 1)
    l = self._res_group(l, 64, 3, 2)
    l = self._res_group(l, 128, 3, 2)
    l = self._bn_relu(l)

    l = self._ASPP(l)

    l = layers.Dropout(0.15)(l)
    l = layers.Conv2D(1, 1, activation='sigmoid')(l)
    l = BilinearUpSampling2D(size=(4, 4))(l)
    self._model = models.Model(inputs=inp, outputs=l)

  def process(self, img):
    img_in = img_get_square(img)
    img_in = cv2.resize(img_in, dsize=self._opt.size, interpolation=cv2.INTER_CUBIC)
    img_src = img_in
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_in = img_in.astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, axis=0)
    img_out = self._model.predict(img_in)
    img_out = img_out[0, :, :, 0]
    img_out *= 255
    img_out = img_out.astype(np.uint8)
    _, img_res = cv2.threshold(img_out, 127, 255, cv2.THRESH_OTSU)
    h, w, c = img_src.shape
    img_show = np.zeros((h, 2*w, c), dtype=np.uint8)
    img_res = np.expand_dims(img_res, -1)
    img_show[:, :w] = img_src
    img_show[:, w:] = img_res
    cv2.imshow('HGR Net', img_show)
    return img_out

