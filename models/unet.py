# unet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import keras

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
        self.affinity_train_step = None

        self.radius = self.config.affinity_radius
        #building mask for sparse pixelwise distances within radius
        #self.radius_mask = np.zeros([
        #    1, #self.config.batch_size,
        #    self.config.image_height * self.config.image_width,
        #    self.config.image_height * self.config.image_width,
        #    self.config.input_channels], dtype=np.float32)
        mask_indices = []
        for i in range(self.config.image_height * self.config.image_width):
            pixi = (i / self.config.image_height, i % self.config.image_width)
            js = np.arange(self.config.image_height * self.config.image_width)
            pixj0 = js % self.config.image_width
            pixj1 = js / self.config.image_height
            dists = np.sqrt(np.square(pixj0 - pixi[0]) + np.square(pixj1 - pixi[1]))
            js = np.argwhere(dists <= self.radius)
            for j in js:
                j = j[0]
                mask_indices.append([0, i, j, 0])
            #for j in range(self.config.image_height * self.config.image_width):
            #    pixj = (j / self.config.image_height, j % self.config.image_width)
            #    pixj = pixj[1], pixj[0]
            #    if np.linalg.norm([pixi, pixj]) <= self.radius:
            #        mask_indices.append([0, i, j, 0]) #self.radius_mask[:,i,j,:] = 1
        self.radius_mask = tf.sparse.SparseTensor(indices=mask_indices, values=[1.0]*len(mask_indices), dense_shape=[1,
            self.config.image_height * self.config.image_width,
            self.config.image_height * self.config.image_width,
            1])
        mask_indices = None
        #self.identity = tf.eye(self.config.image_height * self.config.image_width, dtype=np.float32)
        #self.identity = tf.expand_dims(self.identity, 0)
        #self.identity = tf.tile(self.identity, [self.config.batch_size, 1, 1])
        self.identity = tf.sparse.eye(self.config.image_height * self.config.image_width)


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
        print(pairwise_l1)
        self.radius_mask = tf.sparse.to_dense(self.radius_mask)
        pairwise_l1 *= self.radius_mask
        #pairwise_l1 = tf.sparse.reorder(pairwise_l1)
        #pairwise_l1 = tf.sparse.to_dense(pairwise_l1)
        print(pairwise_l1)
        affinity_conv = Conv2D(filters=1, kernel_size=(1,1), padding='same')(pairwise_l1)
        affinity_conv = keras.activations.exponential(affinity_conv)
        print(affinity_conv)

        #generating pairwise_l1 ground truth
        slim_y = tf.reshape(self.y, [batch_size, mult_dims, self.config.output_channels])
        expanded = tf.tile(tf.expand_dims(slim_y, 2), [1,1,mult_dims,1])
        expanded_t = tf.transpose(expanded, perm=[0,2,1,3])
        affinity_gt = tf.cast(tf.equal(expanded, expanded_t), tf.float32)
        affinity_gt = tf.reduce_prod(affinity_gt, 3, keepdims=True) #if all output channels match, affinity is set to 1
        affinity_gt *= self.radius_mask
        #affinity_gt = tf.sparse.to_dense(affinity_gt)
        self.radius_mask = tf.contrib.layers.dense_to_sparse(self.radius_mask) #return to sparse form to save memory
        ####
        self.euclidean_loss = affinity_gt - affinity_conv
        self.euclidean_loss = K.square(self.euclidean_loss)
        self.euclidean_loss = K.sum(self.euclidean_loss, axis=[1, 2, 3])
        self.euclidean_loss = K.sqrt(self.euclidean_loss)
        print(self.euclidean_loss)
        affinity_conv = tf.squeeze(affinity_conv, 3)
        return affinity_conv

    def random_walk_layer(self, segmentation, affinity, batch_size):
        segmentation = tf.reshape(segmentation, [batch_size, self.config.image_height * self.config.image_width,
                                                 self.config.output_channels])
        walk = None
        if(self.is_evaluating): #walk to convergence
            alpha = self.config.rwn_tradeoff_alpha
            walk = tf.matmul(tf.linalg.inv(self.identity - alpha * affinity), segmentation)
        else: #walk once
            #affinity = tf.contrib.layers.dense_to_sparse(affinity)
            walk = tf.linalg.matmul(affinity, segmentation)#walk = tf.sparse.sparse_dense_matmul(affinity, segmentation)
            #walk = tf.sparse.to_dense(walk)
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
            self.conv2 = e1

        #Affinity branch
        with K.name_scope("Affinity"):
            affinity = self.affinity_branch(batch_size)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.euclidean_loss)

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

        # Random Walk
        self.fn = self.random_walk_layer(self.fn, affinity, batch_size)

        with K.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(categorical_crossentropy(self.y, self.fn))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             self.global_step)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
