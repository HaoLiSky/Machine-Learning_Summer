"""

"""
import sys

import keras.backend as K
import tensorflow as tf

from keras.callbacks import Callback
from keras.engine.topology import Layer


def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def spread(y_true, y_pred):
    return K.min(y_pred) - K.max(y_pred)


def mean_diff(y_true, y_pred):
    return K.mean(y_pred) - K.mean(y_true)


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


class endmask(Layer):
    def __init__(self, **kwargs):
        super(endmask, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = tf.where(tf.is_nan(inputs), K.zeros_like(inputs), inputs)
        return outputs


class LossHistory(Callback):
    def __init__(self):
        self.seen = 0
        self.test_losses = []
        self.train_losses = []
        self.logs = {}

    def on_epoch_end(self, epoch, logs):
        self.train_losses.append(logs.get('loss'))
        self.test_losses.append(logs.get('val_loss'))

        self.seen += 1
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
