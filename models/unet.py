# unet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base.base_model import BaseModel
from keras import backend as K
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, MaxPool2D, ReLU, Softmax
from keras.objectives import categorical_crossentropy


class Unet(BaseModel):
    def __init__(self, config, is_evaluating=False):
        super(Unet, self).__init__(config, is_evaluating)

        self.x = None
        self.y = None
        self.cross_entropy = None
        self.fn = None
        self.train_step = None
        self.gen_filters = self.config.gen_filters

        self.input_shape = tf.TensorShape([self.config.image_height,
                                           self.config.image_width,
                                           self.config.input_channels])

        self.output_shape = tf.TensorShape([self.config.image_height,
                                            self.config.image_width,
                                            self.config.output_channels])

        self.build_model(1 if self.is_evaluating else self.config.batch_size)
        self.init_saver()

    def build_model(self, batch_size):
        self.x = tf.placeholder(tf.float32, shape=[batch_size] + self.input_shape.as_list())
        self.y = tf.placeholder(tf.float32, shape=[batch_size] + self.output_shape.as_list())

        # Network Architecture

        with K.name_scope("Encode1"):
            # First Convolution
            e1 = Conv2D(filters=self.gen_filters, kernel_size=(3, 3), padding='same')(self.x)
            e1 = ReLU()(e1)

            # Second Convolution
            e1 = Conv2D(filters=self.gen_filters, kernel_size=(3, 3), padding='same')(e1)
            e1 = ReLU()(e1)

        # Max Pool
        e2 = MaxPool2D(padding='same')(e1)

        with K.name_scope("Encode2"):
            # First Convolution
            e2 = Conv2D(filters=self.gen_filters * 2, kernel_size=(3, 3), padding='same')(e2)
            e2 = ReLU()(e2)

            # Second Convolution
            e2 = Conv2D(filters=self.gen_filters * 2, kernel_size=(3, 3), padding='same')(e2)
            e2 = ReLU()(e2)

        # Max Pool
        e3 = MaxPool2D(padding='same')(e2)

        with K.name_scope("Encode3"):
            # First Convolution
            e3 = Conv2D(filters=self.gen_filters * 4, kernel_size=(3, 3), padding='same')(e3)
            e3 = ReLU()(e3)

            # Second Convolution
            e3 = Conv2D(filters=self.gen_filters * 4, kernel_size=(3, 3), padding='same')(e3)
            e3 = ReLU()(e3)

        # Max Pool
        e4 = MaxPool2D(padding='same')(e3)

        with K.name_scope("Encode4"):
            # First Convolution
            e4 = Conv2D(filters=self.gen_filters * 8, kernel_size=(3, 3), padding='same')(e4)
            e4 = ReLU()(e4)

            # Second Convolution
            e4 = Conv2D(filters=self.gen_filters * 8, kernel_size=(3, 3), padding='same')(e4)
            e4 = ReLU()(e4)

        # Max Pool
        e5 = MaxPool2D(padding='same')(e4)

        with K.name_scope("Encode5"):
            # First Convolution
            e5 = Conv2D(filters=self.gen_filters * 16, kernel_size=(3, 3), padding='same')(e5)
            e5 = ReLU()(e5)

            # Second Convolution
            e5 = Conv2D(filters=self.gen_filters * 16, kernel_size=(3, 3), padding='same')(e5)
            e5 = ReLU()(e5)

        with K.name_scope("Decode1"):
            # Transpose Convolution
            d1 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=(2, 2), strides=(2, 2), padding='same')(e5)

            # Skip Connection
            d1 = Concatenate()([d1, e4])

            # First Convolution
            d1 = Conv2D(filters=self.gen_filters * 8, kernel_size=(3, 3), padding='same')(d1)
            d1 = ReLU()(d1)

            # Second Convolution
            d1 = Conv2D(filters=self.gen_filters * 8, kernel_size=(3, 3), padding='same')(d1)
            d1 = ReLU()(d1)

        # Dropout
        d1 = Dropout(rate=0.5)(d1)

        with K.name_scope("Decode2"):
            # Transpose Convolution
            d2 = Conv2DTranspose(filters=self.gen_filters * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')(d1)

            # Skip Connection
            d2 = Concatenate()([d2, e3])

            # First Convolution
            d2 = Conv2D(filters=self.gen_filters * 4, kernel_size=(3, 3), padding='same')(d2)
            d2 = ReLU()(d2)

            # Second Convolution
            d2 = Conv2D(filters=self.gen_filters * 4, kernel_size=(3, 3), padding='same')(d2)
            d2 = ReLU()(d2)

        # Dropout
        d2 = Dropout(rate=0.5)(d2)

        with K.name_scope("Decode3"):
            # Transpose Convolution
            d3 = Conv2DTranspose(filters=self.gen_filters * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(d2)

            # Skip Connection
            d3 = Concatenate()([d3, e2])

            # First Convolution
            d3 = Conv2D(filters=self.gen_filters * 2, kernel_size=(3, 3), padding='same')(d3)
            d3 = ReLU()(d3)

            # Second Convolution
            d3 = Conv2D(filters=self.gen_filters * 2, kernel_size=(3, 3), padding='same')(d3)
            d3 = ReLU()(d3)

        with K.name_scope("Decode4"):
            # Transpose Convolution
            d4 = Conv2DTranspose(filters=self.gen_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(d3)

            # Skip Connection
            d4 = Concatenate()([d4, e1])

            # First Convolution
            d4 = Conv2D(filters=self.gen_filters, kernel_size=(3, 3), padding='same')(d4)
            d4 = ReLU()(d4)

            # Second Convolution
            d4 = Conv2D(filters=self.gen_filters, kernel_size=(3, 3), padding='same')(d4)
            d4 = ReLU()(d4)

        # Final Convolution
        self.fn = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=(1, 1), padding='same')(d4)
        self.fn = Softmax()(self.fn)

        with K.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(categorical_crossentropy(self.y, self.fn))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             self.global_step)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
