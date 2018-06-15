"""

"""
import sys

import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects

def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def spread(y_true, y_pred):
    return K.min(y_pred) - K.max(y_pred)


def mean_diff(y_true, y_pred):
    return K.mean(y_pred) - K.mean(y_true)


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def custom_sigmoid(x):
    return (K.sigmoid(x)) - 0.5    #to make sure the padding zero inputs can also output zero after the calculations in activations#

class endmask(Layer):
    def __init__(self, **kwargs):
        super(endmask, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = tf.where(tf.is_nan(inputs), K.zeros_like(inputs), inputs)
        return outputs


class LossTracking(Callback):
    def __init__(self):
        self.seen = 0
        self.test_losses = []
        self.train_losses = []
        self.logs = {}
        self.patience = 1000
        self.overfit_steps = 0
        self.threshold = 0.9
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        self.seen += 1
        current_loss = logs.get('loss')
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

        if (current_loss < current_val_loss * self.threshold
                and current_val_loss > self.best_val_loss):
            self.overfit_steps += 1
            if self.overfit_steps > self.patience:
                print('Stopped early after {} epochs.'.format(self.seen))
                self.model.stop_training = True
        else:
            self.overfit_steps = 0
        self.train_losses.append(current_loss)
        self.test_losses.append(current_val_loss)

        # if self.seen % self.display == 0:
        try:
            print('\r{}:'.format(self.seen).ljust(8),
                  '{0:.3f}'.format(logs.get('loss')).rjust(9),
                  '{0:.3f}'.format(logs.get('mean_pred')).rjust(9),
                  '{0:.3f}'.format(logs.get('spread')).rjust(9),
                  '   |',
                  '{0:.3f}'.format(logs.get('val_loss')).rjust(9),
                  '{0:.3f}'.format(logs.get('val_mean_pred')).rjust(9),
                  '{0:.3f}'.format(logs.get('val_spread')).rjust(9),

                  end='')
            sys.stdout.flush()
        except (TypeError, ValueError):
            pass
