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

import numpy as np
import tensorflow as tf

import util


class Sat2Seg(object):
    """...

    ...
    """

    def __init__(self,
                 image_width=512,
                 image_height=256,
                 genf_dim=64,
                 inpc_dim=3,
                 outc_dim=3):
        """...

        Args:
        :param image_width:
        :param image_height:
        :param genf_dim:
        :param inpc_dim:
        :param outc_dim:
        """
        self._genf_dim = genf_dim
        self._disf_dim = genf_dim

        self._input_shape = tf.TensorShape([image_height, image_width, inpc_dim])
        self._output_shape = tf.TensorShape([image_height, image_width, outc_dim])

    def generator(self, real_son, batch_size, reuse=False):
        """...

        :param real_son:
        :param batch_size:
        :param reuse:

        Returns:
        ...
        """
        with tf.variable_scope("generator", reuse=reuse):
            s = self._output_shape.as_list()[0]
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            r = self._output_shape.as_list()[1]
            r2, r4, r8, r16, r32, r64, r128 = int(r/2), int(r/4), int(r/8), int(r/16), int(r/32), int(r/64), int(r/128)

            # First Encode
            e1, w1 = util.conv2d(real_son, self._genf_dim, name="g_e1_conv", with_w=True)

            # Second Encode
            e2, w2 = util.conv2d(util.lrelu(e1), self._genf_dim * 2, name="g_e2_conv", with_w=True)
            e2 = util.batch_norm(e2, name="g_bn_e2")

            # Third Encode
            e3, w3 = util.conv2d(util.lrelu(e2), self._genf_dim * 4, name="g_e3_conv", with_w=True)
            e3 = util.batch_norm(e3, name="g_bn_e3")

            # Fourth Encode
            e4, w4 = util.conv2d(util.lrelu(e3), self._genf_dim * 8, name="g_e4_conv", with_w=True)
            e4 = util.batch_norm(e4, name="g_bn_e4")

            # Fifth Encode
            e5, w5 = util.conv2d(util.lrelu(e4), self._genf_dim * 8, name="g_e5_conv", with_w=True)
            e5 = util.batch_norm(e5, name="g_bn_e5")

            # Sixth Encode
            e6, w6 = util.conv2d(util.lrelu(e5), self._genf_dim * 8, name="g_e6_conv", with_w=True)
            e6 = util.batch_norm(e6, name="g_bn_e6")

            # Seventh Encode
            e7, w7 = util.conv2d(util.lrelu(e6), self._genf_dim * 8, name="g_e7_conv", with_w=True)
            e7 = util.batch_norm(e7, name="g_bn_e7")

            # Eighth Encode
            e8, w8 = util.conv2d(util.lrelu(e7), self._genf_dim * 8, name="g_e8_conv", with_w=True)
            e8 = util.batch_norm(e8, name="g_bn_e8")

            # First Decode
            d1, w9 = util.deconv2d(tf.nn.relu(e8), [batch_size, s128, r128, self._genf_dim * 8], name="g_d1", with_w=True)
            d1 = tf.nn.dropout(util.batch_norm(d1, name="g_bn_d1"), 0.5)
            d1 = tf.concat([d1, e7], axis=3)

            # Second Decode
            d2, w10 = util.deconv2d(tf.nn.relu(d1), [batch_size, s64, r64, self._genf_dim * 8], name="g_d2", with_w=True)
            d2 = tf.nn.dropout(util.batch_norm(d2, name="g_bn_d2"), 0.5)
            d2 = tf.concat([d2, e6], axis=3)

            # Third Decode
            d3, w11 = util.deconv2d(tf.nn.relu(d2), [batch_size, s32, r32, self._genf_dim * 8], name="g_d3", with_w=True)
            d3 = tf.nn.dropout(util.batch_norm(d3, name="g_bn_d3"), 0.5)
            d3 = tf.concat([d3, e5], axis=3)

            # Fourth Decode
            d4, w12 = util.deconv2d(tf.nn.relu(d3), [batch_size, s16, r16, self._genf_dim * 8], name="g_d4", with_w=True)
            d4 = util.batch_norm(d4, name="g_bn_d4")
            d4 = tf.concat([d4, e4], axis=3)

            # Fifth Decode
            d5, w13 = util.deconv2d(tf.nn.relu(d4), [batch_size, s8, r8, self._genf_dim * 4], name="g_d5", with_w=True)
            d5 = util.batch_norm(d5, name="g_bn_d5")
            d5 = tf.concat([d5, e3], axis=3)

            # Sixth Decode
            d6, w14 = util.deconv2d(tf.nn.relu(d5), [batch_size, s4, r4, self._genf_dim * 2], name="g_d6", with_w=True)
            d6 = util.batch_norm(d6, name="g_bn_d6")
            d6 = tf.concat([d6, e2], axis=3)

            # Seventh Decode
            d7, w15 = util.deconv2d(tf.nn.relu(d6), [batch_size, s2, r2, self._genf_dim], name="g_d7", with_w=True)
            d7 = util.batch_norm(d7, name="g_bn_d7")
            d7 = tf.concat([d7, e1], axis=3)

            # Eighth Decode
            d8, w16 = util.deconv2d(tf.nn.relu(d7), [batch_size, s, r, self._output_shape.as_list()[-1]], name="g_d8", with_w=True)

            # Guided Layers
            normalizer_params = { 'decay': 0.9, 'epsilon': 1e-5, 'updates_collections': None }

            # First Guided
            c1 = tf.contrib.layers.conv2d(inputs=d8, num_outputs=self._output_shape.as_list()[-1], kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params, activation_fn=tf.nn.relu)
            g1 = util.guided_filter(real_son, c1, r=20, eps=10**-6, nhwc=True)

            # Second Guided
            c2 = tf.contrib.layers.conv2d(inputs=d8, num_outputs=self._output_shape.as_list()[-1], kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params, activation_fn=tf.nn.relu)
            g2 = util.guided_filter(real_son, c2, r=20, eps=10**-6, nhwc=True)

            # Guided Concatenation
            gf = tf.concat([g2 * d8, g1, d8], axis=3)

            # Final Convolution
            final = tf.contrib.layers.conv2d(inputs=gf, num_outputs=self._output_shape.as_list()[-1], kernel_size=[1, 1],
                                             stride=[1, 1], padding='SAME',
                                             normalizer_fn=None, activation_fn=None)

            return tf.nn.softmax(final), final, \
                   [tf.abs(tf.reduce_mean(w1)),
                    tf.abs(tf.reduce_mean(w2)),
                    tf.abs(tf.reduce_mean(w3)),
                    tf.abs(tf.reduce_mean(w4)),
                    tf.abs(tf.reduce_mean(w5)),
                    tf.abs(tf.reduce_mean(w6)),
                    tf.abs(tf.reduce_mean(w7)),
                    tf.abs(tf.reduce_mean(w8)),
                    tf.abs(tf.reduce_mean(w9)),
                    tf.abs(tf.reduce_mean(w10)),
                    tf.abs(tf.reduce_mean(w11)),
                    tf.abs(tf.reduce_mean(w12)),
                    tf.abs(tf.reduce_mean(w13)),
                    tf.abs(tf.reduce_mean(w14)),
                    tf.abs(tf.reduce_mean(w15)),
                    tf.abs(tf.reduce_mean(w16))]

    def initial_state(self, batch_size=16, reuse=False):
        """...

        :param batch_size:

        Returns:
        ...
        """
        # Placeholders
        real_son = tf.placeholder(
            name="real_sonar",
            shape=[batch_size] + self._input_shape.as_list(),
            dtype=tf.float32)
        real_img = tf.placeholder(
            name="real_image",
            shape=[batch_size] + self._output_shape.as_list(),
            dtype=tf.float32)

        # Network Outputs
        fake_img, fake_img_logits, weights = self.generator(real_son, batch_size, reuse=reuse)

        # Cost Functions
        img_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=fake_img_logits, labels=real_img)
        )

        # Summaries
        tf.summary.scalar("img_loss", img_loss, collections=[tf.GraphKeys.SUMMARIES])
        tf.summary.image("real_son", real_son, collections=[tf.GraphKeys.SUMMARIES])
        tf.summary.image("fake_img", fake_img, collections=[tf.GraphKeys.SUMMARIES])
        tf.summary.image("real_img", real_img, collections=[tf.GraphKeys.SUMMARIES])

        return img_loss, real_son, real_img

    def final_state(self, batch_size=1):
        """...

        :param batch_size:

        Returns:
        ...
        """
        # Placeholders
        real_son = tf.placeholder(
            name="real_sonar",
            shape=[batch_size] + self._input_shape.as_list(),
            dtype=tf.float32)

        fake_sam, _, weights = self.generator(real_son, batch_size, reuse=tf.AUTO_REUSE)

        return real_son, fake_sam
