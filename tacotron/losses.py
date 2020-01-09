# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf


def l1_loss(y_hat, y, mask):
    loss = tf.abs(y_hat - y)
    return tf.losses.compute_weighted_loss(loss, weights=tf.expand_dims(mask, axis=2))


def mse_loss(y_hat, y, mask):
    loss = tf.losses.mean_squared_error(y, y_hat, weights=tf.expand_dims(mask, axis=2))
    # tf.losses.mean_squared_error cast output to float32 so the output is casted back to the original precision
    if loss.dtype is not y.dtype:
        return tf.cast(loss, dtype=y.dtype)
    else:
        return loss


def spec_loss(y_hat, y, mask, spec_loss_type):
    if spec_loss_type == "l1":
        return l1_loss(y_hat, y, mask)
    elif spec_loss_type == "mse":
        return mse_loss(y_hat, y, mask)
    else:
        raise ValueError(f"Unknown loss type: {spec_loss_type}")


def classification_loss(y_hat, y, mask):
    return tf.losses.softmax_cross_entropy(y, y_hat, weights=mask)


def binary_loss(done_hat, done, mask):
    return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1), weights=mask)
