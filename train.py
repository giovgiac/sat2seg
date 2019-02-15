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

import glob
import os
import time

import numpy as np
import tensorflow as tf

import sat2seg
import util

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("num_gen_filters", 16, "Number of generator filters in the first convolutional layer.")
tf.flags.DEFINE_integer("num_input_channels", 3, "Number of input image channels.")
tf.flags.DEFINE_integer("num_output_channels", 3, "Number of output image channels.")

# Dataset parameters
tf.flags.DEFINE_integer("num_images", 1100, "Number of images in the dataset for training.")
tf.flags.DEFINE_integer("image_width", 256, "Width to scale images during training (input and output).")
tf.flags.DEFINE_integer("image_height", 256, "Height to scale images during training (input and output).")

# Optimizer parameters
tf.flags.DEFINE_float("learning_rate", 0.0002, "Initial learning rate for Adam optimizer.")
tf.flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam optimizer.")

# Task parameters
tf.flags.DEFINE_integer("num_epochs", 100, "Number of epochs to run for.")
tf.flags.DEFINE_integer("batch_size", 128, "Size of the batch to use.")

# Training options
tf.flags.DEFINE_string("mode", "evaluate", "Mode to run, one of: train, restore, evaluate, validate.")
tf.flags.DEFINE_string("evaluate_dir", "./results", "The folder to save the results.")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "The folder to save the model.")
tf.flags.DEFINE_integer("checkpoint_interval", 100, "The number of steps between savings of the checkpoint.")
tf.flags.DEFINE_string("summary_dir", "./log", "The folder to save the summaries.")
tf.flags.DEFINE_integer("summary_interval", 5, "The number of steps between saving the summaries.")
tf.flags.DEFINE_integer("report_interval", 1, "The number of steps between reporting loss and other info.")


def train(num_epochs, report_interval):
    model = sat2seg.Sat2Seg(
        image_width=FLAGS.image_width,
        image_height=FLAGS.image_height,
        genf_dim=FLAGS.num_gen_filters,
        inpc_dim=FLAGS.num_input_channels,
        outc_dim=FLAGS.num_output_channels)

    img_loss, real_son, real_img = model.initial_state(FLAGS.batch_size)

    # Obtain variables for optimization
    vars = tf.trainable_variables()
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    '''
    errors = []
    test_loss = np.mean(errors)
    with tf.variable_scope("generator", reuse=True):
        tf.summary.scalar("test_img_loss", test_loss, collections=[tf.GraphKeys.SUMMARIES])
    '''

    # Configure optimizers for network
    g_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                  beta1=FLAGS.beta1).minimize(img_loss, global_step=global_step, var_list=vars)
    sum = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    saver = tf.train.Saver()
    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.checkpoint_dir,
                save_steps=FLAGS.checkpoint_interval,
                saver=saver),
            tf.train.SummarySaverHook(
                output_dir=FLAGS.summary_dir,
                save_steps=FLAGS.summary_interval,
                summary_op=[sum]
            )
        ]
    else:
        hooks = []

    data_son = sorted(glob.glob("./datasets/train/input/*.png"))
    data_img = sorted(glob.glob("./datasets/train/gt/*.png"))

    #data_son_val = sorted(glob.glob("./datasets/test/input/*.png"))
    #data_rea_val = sorted(glob.glob("./datasets/test/gt/*.png"))

    # Begin Training
    with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
        # Prepare counting
        counter = 0

        for epoch in range(FLAGS.num_epochs):
            batch_idxs = min(len(data_son), FLAGS.num_images) // FLAGS.batch_size
            for idx in range(batch_idxs):
                # Get files for images
                batch_files_son = data_son[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                batch_files_img = data_img[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]

                # Get the actual images
                batch_son = [util.load_data(batch_file, is_sonar=False) for batch_file in batch_files_son]
                batch_son = np.array(batch_son).astype(np.float32)
                batch_img = [util.load_data(batch_file, is_sonar=False) for batch_file in batch_files_img]
                batch_img = np.array(batch_img).astype(np.float32)

                # Update generator network
                _, g_err, step = sess.run([g_op, img_loss, global_step], feed_dict={real_son: batch_son, real_img: batch_img})

                counter += 1
                if counter % report_interval == 0:
                    tf.logging.info("RESULT: Epoch %2d: [%4d/%4d] step %d, g_loss %.8f"
                                    % (epoch + 1, idx + 1, batch_idxs + 1, step, g_err))

            # Validation Step
            '''
            if np.mod(epoch, 2) == 0:
                data = [util.load_data(file, is_sonar=False) for file in data_son_val]
                data = np.array(data).astype(np.float32)
                data = [data[i:i + FLAGS.batch_size] for i in range(0, len(data), FLAGS.batch_size)]
                data = np.array(data)

                real = [util.load_data(file, is_sonar=False) for file in data_rea_val]
                real = np.array(real).astype(np.float32)
                real = [real[i:i + FLAGS.batch_size] for i in range(0, len(real), FLAGS.batch_size)]
                real = np.array(real)

                for i in range(len(data)):
                    err = sess.run([img_loss], feed_dict={real_son: data[i], real_img: real[i]})
                    errors.append(err)

                test_loss = np.mean(errors)
                tf.logging.info("VALIDATION: Loss: %.8f" % (test_loss))

                errors.clear()
            '''



def evaluate(checkpoint_dir, evaluate_dir, batch_size=1):
    # Create the model
    model = sat2seg.Sat2Seg(
        image_width=FLAGS.image_width,
        image_height=FLAGS.image_height,
        genf_dim=FLAGS.num_gen_filters,
        inpc_dim=FLAGS.num_input_channels,
        outc_dim=FLAGS.num_output_channels)

    real_son, fake_sam = model.final_state(batch_size)

    saver = tf.train.Saver()

    # Run the evaluation
    with tf.Session() as sess:
        data_son = sorted(glob.glob("./datasets/test/input/*.png"))

        data = [util.load_data(file, is_sonar=False) for file in data_son]
        data = np.array(data).astype(np.float32)
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        data = np.array(data)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            tf.logging.info("EVALUATING: Checkpoint restored")
        else:
            tf.logging.error("Checkpoint not found")
            return

        for i, img in enumerate(data):
            idx = i

            tf.logging.info("EVALUATING: Batch %d" % idx)
            evaluates = sess.run(fake_sam, feed_dict={real_son: img})
            util.save_images(evaluates, batch_size, evaluate_dir, idx)


def validate(checkpoint_dir, batch_size=1):
    # Create the model
    model = sat2seg.Sat2Seg(
        image_width=FLAGS.image_width,
        image_height=FLAGS.image_height,
        genf_dim=FLAGS.num_gen_filters,
        inpc_dim=FLAGS.num_input_channels,
        outc_dim=FLAGS.num_output_channels)

    img_loss, _, real_son, real_img = model.initial_state(batch_size)

    saver = tf.train.Saver()

    # Run validation
    with tf.Session() as sess:
        data_son = sorted(glob.glob("./datasets/test/input/*.png"))
        data_rea = sorted(glob.glob("./datasets/test/gt/*.png"))

        data = [util.load_data(file, is_sonar=False) for file in data_son]
        data = np.array(data).astype(np.float32)
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        data = np.array(data)

        real = [util.load_data(file, is_sonar=False) for file in data_rea]
        real = np.array(real).astype(np.float32)
        real = [real[i:i+batch_size] for i in range(0, len(real), batch_size)]
        real = np.array(real)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            tf.logging.info("VALIDATING: Checkpoint restored")
        else:
            tf.logging.error("Checkpoint not found")
            return

        assert(len(data) == len(real))
        errors = []
        for i in range(len(data)):
            err = sess.run([img_loss], feed_dict={real_son: data[i], real_img: real[i]})
            errors.append(err)

        tf.logging.info("VALIDATION: Loss: %.8f" % (np.mean(errors)))


def main(unused_argv):
    tf.logging.set_verbosity(3)

    if FLAGS.mode == "train":
        train(FLAGS.num_epochs, FLAGS.report_interval)
    elif FLAGS.mode == "restore":
        pass
    elif FLAGS.mode == "evaluate":
        evaluate(FLAGS.checkpoint_dir, FLAGS.evaluate_dir)
    elif FLAGS.mode == "validate":
        validate(FLAGS.checkpoint_dir)
    else:
        tf.logging.error("Unknown mode chosen for network, please choose one of: train, restore, evaluate.\n")


if __name__ == '__main__':
    tf.app.run()
