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
    LSTMBlockFusedCell = "tf.contrib.rnn.LSTMBlockFusedCell"
    CudnnLSTM = "tf.contrib.cudnn_rnn.CudnnLSTM"
    CudnnCompatibleLSTM = "tf.contrib.cudnn_rnn.CudnnCompatibleLSTM"

    all_list = [LSTMCell, LSTMBlockCell, LSTMBlockFusedCell, CudnnLSTM, CudnnCompatibleLSTM]


class GRUImpl:
    GRUCell = "tf.contrib.rnn.GRUCell"
    GRUBlockCellV2 = "tf.contrib.rnn.GRUBlockCellV2"
    CudnnGRU = "tf.contrib.cudnn_rnn.CudnnGRU"
    CudnnCompatibleGRUCell = "tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell"

    all_list = [GRUCell, GRUBlockCellV2, CudnnGRU, CudnnCompatibleGRUCell]


def lstm_factory(lstm_impl, num_layers, num_units):
    if lstm_impl == LSTMImpl.LSTMCell:
        cell = tf.nn.rnn_cell.LSTMCell(num_units) if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(num_layers)])
        return cell
    elif lstm_impl == LSTMImpl.LSTMBlockCell:
        cell = tf.contrib.rnn.LSTMBlockCell(num_units) if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.LSTMBlockCell(num_units) for _ in range(num_layers)])
        return cell
    elif lstm_impl == LSTMImpl.LSTMBlockFusedCell:
        cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units) if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.LSTMBlockFusedCell(num_units) for _ in range(num_layers)])
        return cell
    elif lstm_impl == LSTMImpl.CudnnLSTM:
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, num_units)
        return cell
    elif lstm_impl == LSTMImpl.CudnnCompatibleLSTM:
        # Even if there's only one layer, the cell needs to be wrapped in MultiRNNCell.
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.cudnn_rnn.CudnnCompatibleLSTM(num_units) for _ in range(num_layers)])
        return cell
    else:
        raise ValueError(f"Unknown LSTM cell implementation: {lstm_impl}. Supported: {', '.join(LSTMImpl.all_list)}")


def gru_factory(gru_impl, num_layers, num_units):
    if gru_impl == GRUImpl.GRUCell:
        cell = tf.nn.rnn_cell.GRUCell(num_units) if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(num_units) for _ in range(num_layers)])
        return cell
    elif gru_impl == GRUImpl.GRUBlockCellV2:
        cell = tf.contrib.rnn.GRUBlockCellV2(num_units) if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.GRUBlockCellV2(num_units) for _ in range(num_layers)])
        return cell
    elif gru_impl == GRUImpl.CudnnGRU:
        cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_units, input_size=num_units)
        return cell
    elif gru_impl == GRUImpl.CudnnCompatibleGRUCell:
        # Even if there's only one layer, the cell needs to be wrapped in MultiRNNCell.
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units) for _ in range(num_layers)])
        return cell
    else:
        raise ValueError(f"Unknown GRU cell implementation: {gru_impl}. Supported: {', '.join(GRUImpl.all_list)}")
