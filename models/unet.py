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
        self.euclidean_loss = None
        self.fn = None
        self.train_step = None

        self.radius = self.config.affinity_radius
        #building mask for sparse pixelwise distances within radius
        self.radius_mask = np.zeros([
            self.config.batch_size,
            self.config.image_height * self.config.image_width,
            self.config.image_height * self.config.image_width,
            self.config.input_channels], dtype=np.float32)
        self.identity = np.identity(self.config.image_height * self.config.image_width, dtype=np.float32)
        self.identity = np.expand_dims(self.identity, 0)
        self.identity = np.tile(self.identity, [batch_size, 1, 1])
        self.identity = tf.constant(self.identity)


        for i in range(self.config.image_height * self.config.image_width):
            pixi = (i / self.config.image_height, i % self.config.image_width)
            for j in range(self.config.image_height * self.config.image_width):
                pixj = (j / self.config.image_height, j % self.config.image_width)
                pixj = pixj[1], pixj[0]
                if ((pixj[0] - pix[0])**2 + (pixj[1] -pix[1])**2)**.5 <= self.radius:
                    self.radius_mask[:i,j,:] = 1

        #first layers for affinity                 
        self.conv1 = None
        self.conv2 = None

        self.gen_filters = self.config.gen_filters

        self.input_shape = tf.TensorShape([self.config.image_height,
                                           self.config.image_width,
                                           self.config.input_channels])

        self.output_shape = tf.TensorShape([self.config.image_height,
                                            self.config.image_width,
                                            self.config.output_channels])

        self.build_model(1 if self.is_evaluating else self.config.batch_size)
        self.init_saver()

    def affinity_branch(self, batch_size):
        stacked = tf.concat([self.x, self.conv1, self.conv2], axis=3)
        mult_dims = self.config.image_width * self.config.image_height
        slim_stacked = tf.reshape(stacked, [batch_size, mult_dims, self.config.input_channels + 2*self.config.gen_filters])
        expanded = tf.tile(tf.expand_dims(slim_stacked, 2), [1,1,mult_dims,1])
        expanded_t = tf.transpose(expanded, perm=[0,2,1,3])
        pairwise_l1 = tf.abs(expanded - expanded_t)
        pairwise_l1 *= self.radius_mask
        affinity_conv = Conv2D(filters=1, kernel_size=(1,1), padding='same')(pairwise_l1)
        affinity_conv = keras.activations.exponential(affinity_conv)

        #generating pairwise_l1 ground truth
        slim_y = tf.reshape(self.y, [batch_size, mult_dims, self.output_channels])
        expanded = tf.tile(tf.expand_dims(slim_y, 2), [1,1,mult_dims,1])
        expanded_t = tf.transpose(expanded, perm=[0,2,1,3])
        affinity_gt = (expanded == expanded_t)
        affinity_gt = tf.reduce_prod(affinity_gt, 3) #if all output channels match, affinity is set to 1
        affinity_gt *= self.radius_mask
        ####

        self.euclidean_loss = K.sqrt(K.sum(K.square(affinity_gt - affinity_conv), axis=-1))
        return affinity_conv

    def random_walk_layer(self, segmentation, affinity, batch_size):
        segmentation = tf.reshape(segmentation, [batch_size, self.image_height * self.image_width,
                                                 self.output_channels])
        walk = None
        if(self.is_evaluating): #walk to convergence
            alpha = self.config.rwn_tradeoff_alpha
            walk = tf.matmul(tf.linalg.inv(self.identity - alpha * affinity), segmentation)
        else: #walk once
            affinity = tf.contrib.layers.dense_to_sparse(affinity)
            walk = tf.sparse.sparse_dense_matmul(affinity, segmentation)
            walk = tf.sparse.to_dense(walk)
        walk = tf.reshape(walk, [batch_size] + self.output_shape.as_list())
        return walk


    def build_model(self, batch_size):
        self.x = tf.placeholder(tf.float32, shape=[batch_size] + self.input_shape.as_list())
        self.y = tf.placeholder(tf.float32, shape=[batch_size] + self.output_shape.as_list())

        # Network Architecture

        with K.name_scope("Encode1"):
            # First Convolution
            e1 = Conv2D(filters=self.gen_filters, kernel_size=(3, 3), padding='same')(self.x)
            e1 = ReLU()(e1)
            self.conv1 = e1

            # Second Convolution
            e1 = Conv2D(filters=self.gen_filters, kernel_size=(3, 3), padding='same')(e1)
            e1 = ReLU()(e1)
            self.conv2 = e2

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
