# aracati.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import tensorflow as tf

from matplotlib.pyplot import imread, imsave
from scipy.misc import imresize


class Aracati(object):
    def __init__(self, config):
        self.config = config
        self.idx = 0

        # Load Data
        self.sat_data = np.array(sorted(glob.glob("./datasets/aracati/train/input/*.png")))
        self.seg_data = np.array(sorted(glob.glob("./datasets/aracati/train/gt/*.png")))

        self.sat_data_val = np.array(sorted(glob.glob("./datasets/aracati/validation/input/*.png")))
        self.seg_data_val = np.array(sorted(glob.glob("./datasets/aracati/validation/gt/*.png")))

        if len(self.sat_data) != len(self.seg_data):
            tf.logging.error("Dataset has unequal number of satellite and segmentation training images")
            raise ValueError
        elif len(self.sat_data_val) != len(self.seg_data_val):
            tf.logging.error("Dataset has unequal number of satellite and segmentation validation images")
            raise ValueError
        else:
            self.num_images = len(self.sat_data)
            self.num_images_val = len(self.sat_data_val)

    def next_batch(self, batch_size, is_test=False):
        batch_idxs = self.num_images_val // batch_size if is_test else self.num_images // batch_size
        if self.idx == batch_idxs:
            self.idx = 0

        if is_test:
            yield [self.load_data(file, is_grayscale=False) for file in self.sat_data_val[self.idx*batch_size:(self.idx+1)*batch_size]], \
                  [self.load_data(file, is_grayscale=False) for file in self.seg_data_val[self.idx*batch_size:(self.idx+1)*batch_size]]
        else:
            yield [self.load_data(file, is_grayscale=False) for file in self.sat_data[self.idx*batch_size:(self.idx+1)*batch_size]], \
                  [self.load_data(file, is_grayscale=False) for file in self.seg_data[self.idx*batch_size:(self.idx+1)*batch_size]]

    @staticmethod
    def load_data(path, width=256, height=256, is_grayscale=False):
        data = imread(path).astype(np.float)
        data = imresize(data, [height, width])
        data = data / 255.0

        if is_grayscale:
            return data[:, :, :1]
        else:
            return data[:, :, :3]

    @staticmethod
    def save_data(path, idx, data):
        imsave("{}/test_{:04d}.png".format(path, idx), data[0])

