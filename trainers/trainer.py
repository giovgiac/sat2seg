# trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random as rand
import tensorflow as tf

from base.base_trainer import BaseTrainer
from keras import backend as K
from tqdm import tqdm


class Trainer(BaseTrainer):
    def __init__(self, config, data, logger, model, session):
        super(Trainer, self).__init__(config, data, logger, model, session)

    def evaluate_data(self, data, save_fn):
        loop = tqdm(range(len(data)))
        idx = 0

        for _ in loop:
            loop.set_description("Evaluating Image [{}/{}]".format(idx, len(data)))

            feed_dict = {self.model.x: data[idx], K.learning_phase(): 0}
            result = self.session.run(self.model.fn, feed_dict=feed_dict)

            save_fn(self.config.evaluate_dir, idx, result)
            idx += 1

    def train_epoch(self):
        loop = tqdm(range(self.data.num_images // self.config.batch_size))
        loop.set_description("Training Epoch [{}/{}]".format(self.model.epoch.eval(self.session),
                                                             self.config.num_epochs))

        err_list = []
        for _ in loop:
            err = self.train_step()

            # Append Data
            err_list.append(err)

            self.data.idx += 1
        self.data.idx = 0

        it = self.model.global_step.eval(self.session)
        summaries_dict = {
            "total_loss": np.mean(err_list)
        }

        self.logger.summarize(it, summarizer="train", summaries_dict=summaries_dict)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, is_validation=False))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, K.learning_phase(): 1}

        _, err = self.session.run([self.model.train_step,
                                   self.model.cross_entropy],
                                  feed_dict=feed_dict)
        return err

    def validate_epoch(self):
        loop = tqdm(range(self.data.num_images_val // self.config.batch_size))
        loop.set_description("Validating Epoch {}".format(self.model.epoch.eval(self.session)))

        err_list, fn_list, y_list, x_list = [], [], [], []
        for _ in loop:
            err, fn, y, x = self.validate_step()

            # Append Data
            err_list.append(err)
            fn_list.append(fn)
            y_list.append(y)
            x_list.append(x)

            self.data.idx += 1
        self.data.idx = 0

        batch = rand.choice(range(len(fn_list)))
        it = self.model.global_step.eval(self.session)
        summaries_dict = {
            "generated": fn_list[batch],
            "satellite": x_list[batch],
            "segmentation": y_list[batch],
            "total_loss": np.mean(err_list),
        }

        self.logger.summarize(it, summarizer="validation", summaries_dict=summaries_dict)
        self.model.save(self.session)

    def validate_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, is_validation=True))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, K.learning_phase(): 0}

        err, fn, y, x = self.session.run([self.model.cross_entropy,
                                          self.model.fn,
                                          self.model.y,
                                          self.model.x],
                                         feed_dict=feed_dict)
        return err, fn, y, x
