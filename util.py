# Copyright 2018 Giovanni Giacomo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.misc

import numpy as np
import tensorflow as tf

from matplotlib.pyplot import imread, imsave


def load_data(path, image_width=256, image_height=256, flip=True, is_sonar=True):
    """...

    :param path:
    :param image_width:
    :param image_height:
    :param flip:
    :param is_sonar:

    Returns:
    ...
    """
    img = load_image(path)
    img = scipy.misc.imresize(img, [image_height, image_width])

    if is_sonar:
        img = np.expand_dims(img, 2)
    img = img / 255.0

    if not is_sonar:
        return img[:,:,:3]
    else:
        return img


def load_image(path):
    """...

    :param path:

    Returns:
    ...
    """
    img = imread(path).astype(np.float)
    return img


def merge(images, size):
    """...

    :param images:
    :param size:

    Returns:
    ...
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def save_images(images, size, dir, id):
    """...

    :param images:
    :param size:
    :param image_path:

    Returns:
    ...
    """
    for i in range(0, size):
        path = "{}/test_{:04d}.png".format(dir, id * size + i)
        imsave(path, images[i])


def batch_norm(x,
               epsilon=1e-5,
               momentum=0.9,
               trainable=True,
               name="batch_norm"):
    """...

    :param x:
    :param epsilon:
    :param momentum:
    :param trainable:
    :param name:

    Returns:
    ...
    """
    return tf.contrib.layers.batch_norm(
        x,
        decay=momentum,
        updates_collections=None,
        epsilon=epsilon,
        scale=True,
        scope=name)


def conv2d(x,
           out_dim,
           k_h=5,
           k_w=5,
           d_h=2,
           d_w=2,
           sd=0.02,
           with_w=False,
           name="conv2d"):
    """...

    :param x:
    :param out_dim:
    :param k_h:
    :param k_w:
    :param d_h:
    :param d_w:
    :param sd:
    :param name:

    Returns:
    ...
    """
    with tf.variable_scope(name):
        #weight = tf.get_variable(
        #    name='w',
        #    shape=[k_h, k_w, x.get_shape()[-1], out_dim],
        #    initializer=tf.truncated_normal_initializer(stddev=sd))
        weight = tf.get_variable(
            name='w',
            shape=[k_h, k_w, x.get_shape()[-1], out_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        bias = tf.get_variable(
            name='b',
            shape=[out_dim],
            initializer=tf.zeros_initializer())

        conv = tf.nn.conv2d(x, weight, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

        if with_w:
            return conv, weight
        else:
            return conv


def deconv2d(x,
             out_dim,
             k_h=5,
             k_w=5,
             d_h=2,
             d_w=2,
             sd=0.02,
             with_w=False,
             name="deconv2d"):
    """...

    :param x:
    :param out_dim:
    :param k_h:
    :param k_w:
    :param d_h:
    :param d_w:
    :param sd:
    :param with_w:
    :param name:

    Returns:
    ...
    """
    with tf.variable_scope(name):
        #weight = tf.get_variable(
        #    name='w',
        #    shape=[k_h, k_w, out_dim[-1], x.get_shape()[-1]],
        #    initializer=tf.random_normal_initializer(stddev=sd))
        weight = tf.get_variable(
            name='w',
            shape=[k_h, k_w, out_dim[-1], x.get_shape()[-1]],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        bias = tf.get_variable(
            name='b',
            shape=[out_dim[-1]],
            initializer=tf.zeros_initializer())

        deconv = tf.nn.conv2d_transpose(x, weight, output_shape=out_dim, strides=[1, d_h, d_w, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

        if with_w:
            return deconv, weight
        else:
            return deconv


def lrelu(x,
          alpha=0.2,
          name="lrelu"):
    """...

    :param x:
    :param alpha:
    :param name:

    Returns:
    ...
    """
    return tf.maximum(
        name=name,
        x=x,
        y=alpha * x)


def linear(x,
           out_size,
           scope=None,
           sd=0.02,
           bias_start=0.0,
           with_w=False):
    """...

    :param x:
    :param out_size:
    :param scope:
    :param sd:
    :param bias_start:
    :param with_w:

    Returns:
    ...
    """
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        mat = tf.get_variable(
            name="matrix",
            shape=[shape[1], out_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=sd))
        bias = tf.get_variable(
            name="bias",
            shape=[out_size],
            initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(x, mat) + bias, mat, bias
        else:
            return tf.matmul(x, mat)


def diff_x(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=2)

    return output


def diff_y(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=3)

    return output


def box_filter(x, r):
    assert x.shape.ndims == 4

    return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=2), r), axis=3), r)


def guided_filter(x, y, r, eps=1e-8, nhwc=False):
    assert x.shape.ndims == 4 and y.shape.ndims == 4

    # data format
    if nhwc:
        x = tf.transpose(x, [0, 3, 1, 2])
        y = tf.transpose(y, [0, 3, 1, 2])

    # shape check
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    assets = [tf.assert_equal(   x_shape[0],  y_shape[0]),
              tf.assert_equal(  x_shape[2:], y_shape[2:]),
              tf.assert_greater(x_shape[2:],   2 * r + 1),
              tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
                                      tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]

    with tf.control_dependencies(assets):
        x = tf.identity(x)

    # N
    N = box_filter(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

    # mean_x
    mean_x = box_filter(x, r) / N
    # mean_y
    mean_y = box_filter(y, r) / N
    # cov_xy
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    if nhwc:
        output = tf.transpose(output, [0, 2, 3, 1])

    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r, eps=1e-8, nhwc=False):
    assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4

    # data format
    if nhwc:
        lr_x = tf.transpose(lr_x, [0, 3, 1, 2])
        lr_y = tf.transpose(lr_y, [0, 3, 1, 2])
        hr_x = tf.transpose(hr_x, [0, 3, 1, 2])

    # shape check
    lr_x_shape = tf.shape(lr_x)
    lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)

    assets = [tf.assert_equal(   lr_x_shape[0], lr_y_shape[0]),
              tf.assert_equal(   lr_x_shape[0], hr_x_shape[0]),
              tf.assert_equal(   lr_x_shape[1], hr_x_shape[1]),
              tf.assert_equal(  lr_x_shape[2:], lr_y_shape[2:]),
              tf.assert_greater(lr_x_shape[2:], 2 * r + 1),
              tf.Assert(tf.logical_or(tf.equal(lr_x_shape[1], 1),
                                      tf.equal(lr_x_shape[1], lr_y_shape[1])), [lr_x_shape, lr_y_shape])]

    with tf.control_dependencies(assets):
        lr_x = tf.identity(lr_x)

    # N
    N = box_filter(tf.ones((1, 1, lr_x_shape[2], lr_x_shape[3]), dtype=lr_x.dtype), r)

    # mean_x
    mean_x = box_filter(lr_x, r) / N
    # mean_y
    mean_y = box_filter(lr_y, r) / N
    # cov_xy
    cov_xy = box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    # mean_A; mean_b
    A    = tf.transpose(A,    [0, 2, 3, 1])
    b    = tf.transpose(b,    [0, 2, 3, 1])
    hr_x = tf.transpose(hr_x, [0, 2, 3, 1])

    mean_A = tf.image.resize_images(A, hr_x_shape[2:])
    mean_b = tf.image.resize_images(b, hr_x_shape[2:])

    output = mean_A * hr_x + mean_b

    if not nhwc:
        output = tf.transpose(output, [0, 3, 1, 2])

    return output