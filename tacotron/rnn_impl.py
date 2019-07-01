# ==============================================================================
# Copyright (c) 2018-2019, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf


class LSTMImpl:
    LSTMCell = "tf.nn.rnn_cell.LSTMCell"
    LSTMBlockCell = "tf.contrib.rnn.LSTMBlockCell"

    all_list = [LSTMCell, LSTMBlockCell]


class GRUImpl:
    GRUCell = "tf.contrib.rnn.GRUCell"
    GRUBlockCellV2 = "tf.contrib.rnn.GRUBlockCellV2"

    all_list = [GRUCell, GRUBlockCellV2]


def lstm_cell_factory(lstm_impl, num_units, dtype=None):
    if lstm_impl == LSTMImpl.LSTMCell:
        cell = tf.nn.rnn_cell.LSTMCell(num_units, dtype=dtype)
        return cell
    elif lstm_impl == LSTMImpl.LSTMBlockCell:
        cell = tf.contrib.rnn.LSTMBlockCell(num_units, dtype=dtype)
        return cell
    else:
        raise ValueError(f"Unknown LSTM cell implementation: {lstm_impl}. Supported: {', '.join(LSTMImpl.all_list)}")


def gru_cell_factory(gru_impl, num_units, dtype=None):
    if gru_impl == GRUImpl.GRUCell:
        cell = tf.nn.rnn_cell.GRUCell(num_units, dtype=dtype)
        return cell
    elif gru_impl == GRUImpl.GRUBlockCellV2:
        cell = tf.contrib.rnn.GRUBlockCellV2(num_units, dtype=dtype)
        return cell
    else:
        raise ValueError(f"Unknown GRU cell implementation: {gru_impl}. Supported: {', '.join(GRUImpl.all_list)}")
