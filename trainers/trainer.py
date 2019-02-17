# trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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
        errs = []

        for _ in loop:
            err = self.train_step()
            errs.append(err)

            loop.set_description("Epoch [{}/{}] -- loss {:.6f}".format(self.model.epoch.eval(self.session),
                                                                       self.config.num_epochs,
                                                                       err))
        err = np.mean(errs)

        batch_x_val, batch_y_val = next(self.data.next_batch(self.config.batch_size, is_test=True))
        feed_dict = {self.model.x: batch_x_val, self.model.y: batch_y_val, K.learning_phase(): 0}
        err_val, fake, real, sat = self.session.run([self.model.cross_entropy,
                                                     self.model.fn, self.model.y, self.model.x], feed_dict=feed_dict)

        it = self.model.global_step.eval(self.session)

        summaries_dict = {
            "train_loss": err,
        }

        summaries_dict_val = {
            "fake": fake,
            "real": real,
            "satellite": sat,
            "validation_loss": err_val
        }

        self.logger.summarize(it, summarizer="train", summaries_dict=summaries_dict)
        self.logger.summarize(it, summarizer="validation", summaries_dict=summaries_dict_val)
        self.model.save(self.session)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, K.learning_phase(): 1}

        _, err= self.session.run([self.model.train_step, self.model.cross_entropy], feed_dict=feed_dict)
        return err
