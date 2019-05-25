# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Hooks. """

import tensorflow as tf
from tensorflow.python.lib.io import file_io
import os
import numpy as np
import re
from typing import List
from util.tfrecord import write_tfrecord, int64_feature, bytes_feature
from util.metrics import plot_alignment, plot_mel, plot_spec


def write_training_result(global_step: int, id: List[int], text: List[str], predicted_mel: List[np.ndarray],
                          ground_truth_mel: List[np.ndarray], mel_length: List[int], alignment: List[np.ndarray],
                          filename: str):
    batch_size = len(ground_truth_mel)
    raw_predicted_mel = [m.tostring() for m in predicted_mel]
    raw_ground_truth_mel = [m.tostring() for m in ground_truth_mel]
    mel_width = ground_truth_mel[0].shape[1]
    padded_mel_length = [m.shape[0] for m in ground_truth_mel]
    predicted_mel_length = [m.shape[0] for m in predicted_mel]
    raw_alignment = [a.tostring() for a in alignment]
    alignment_source_length = [a.shape[1] for a in alignment]
    alignment_target_length = [a.shape[2] for a in alignment]
    example = tf.train.Example(features=tf.train.Features(feature={
        'global_step': int64_feature([global_step]),
        'batch_size': int64_feature([batch_size]),
        'id': int64_feature(id),
        'text': bytes_feature(text),
        'predicted_mel': bytes_feature(raw_predicted_mel),
        'ground_truth_mel': bytes_feature(raw_ground_truth_mel),
        'mel_length': int64_feature(padded_mel_length),
        'mel_length_without_padding': int64_feature(mel_length),
        'predicted_mel_length': int64_feature(predicted_mel_length),
        'mel_width': int64_feature([mel_width]),
        'alignment': bytes_feature(raw_alignment),
        'alignment_source_length': int64_feature(alignment_source_length),
        'alignment_target_length': int64_feature(alignment_target_length),
    }))
    write_tfrecord(example, filename)


def write_postnet_training_result(global_step: int, ids: List[str], predicted_spec: List[np.ndarray],
                                  ground_truth_spec: List[np.ndarray], spec_length: List[int],
                                  filename: str):
    batch_size = len(ground_truth_spec)
    raw_predicted_spec = [m.tostring() for m in predicted_spec]
    raw_ground_truth_spec = [m.tostring() for m in ground_truth_spec]
    spec_width = ground_truth_spec[0].shape[1]
    padded_spec_length = [m.shape[0] for m in ground_truth_spec]
    predicted_spec_length = [m.shape[0] for m in predicted_spec]
    ids_bytes = [s.encode("utf-8") for s in ids]
    example = tf.train.Example(features=tf.train.Features(feature={
        'global_step': int64_feature([global_step]),
        'batch_size': int64_feature([batch_size]),
        'id': bytes_feature(ids_bytes),
        'predicted_spec': bytes_feature(raw_predicted_spec),
        'ground_truth_spec': bytes_feature(raw_ground_truth_spec),
        'spec_length': int64_feature(padded_spec_length),
        'spec_length_without_padding': int64_feature(spec_length),
        'predicted_spec_length': int64_feature(predicted_spec_length),
        'spec_width': int64_feature([spec_width]),
    }))
    write_tfrecord(example, filename)


class MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, alignment_tensors, global_step_tensor, predicted_mel_tensor, ground_truth_mel_tensor,
                 mel_length_tensor, id_tensor,
                 text_tensor, save_steps,
                 mode, writer: tf.summary.FileWriter,
                 save_training_time_metrics=True,
                 keep_eval_results_max_epoch=10):
        self.alignment_tensors = alignment_tensors
        self.global_step_tensor = global_step_tensor
        self.predicted_mel_tensor = predicted_mel_tensor
        self.ground_truth_mel_tensor = ground_truth_mel_tensor
        self.mel_length_tensor = mel_length_tensor
        self.id_tensor = id_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer
        self.save_training_time_metrics = save_training_time_metrics
        self.keep_eval_results_max_epoch = keep_eval_results_max_epoch
        self.checkpoint_pattern = re.compile('all_model_checkpoint_paths: "model.ckpt-(\d+)"')

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, alignments, predicted_mels, ground_truth_mels, mel_length, ids, texts = run_context.session.run(
                (self.global_step_tensor, self.alignment_tensors, self.predicted_mel_tensor,
                 self.ground_truth_mel_tensor, self.mel_length_tensor, self.id_tensor, self.text_tensor))
            alignments = [a.astype(np.float32) for a in alignments]
            predicted_mels = [m.astype(np.float32) for m in list(predicted_mels)]
            ground_truth_mels = [m.astype(np.float32) for m in list(ground_truth_mels)]
            if self.mode == tf.estimator.ModeKeys.EVAL or self.save_training_time_metrics:
                id_strings = ",".join([str(i) for i in ids][:10])
                result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
                tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)
                write_training_result(global_step_value, list(ids), list(texts), predicted_mels,
                                      ground_truth_mels, list(mel_length),
                                      alignments,
                                      filename=os.path.join(self.writer.get_logdir(), result_filename))
            if self.mode == tf.estimator.ModeKeys.EVAL:
                alignments = [[a[i] for a in alignments] for i in range(alignments[0].shape[0])]
                for _id, text, align, pred_mel, gt_mel in zip(ids, texts, alignments, predicted_mels,
                                                              ground_truth_mels):
                    output_filename = "{}_result_step{:09d}_{:d}.png".format(self.mode,
                                                                             global_step_value, _id)
                    plot_alignment(align, text.decode('utf-8'), _id, global_step_value,
                                   os.path.join(self.writer.get_logdir(), "alignment_" + output_filename))
                    plot_mel(gt_mel, pred_mel, text.decode('utf-8'), _id, global_step_value,
                             os.path.join(self.writer.get_logdir(), "mel_" + output_filename))

    def end(self, session):
        if self.mode == tf.estimator.ModeKeys.EVAL:
            current_global_step = session.run(self.global_step_tensor)
            with open(os.path.join(self.writer.get_logdir(), "checkpoint")) as f:
                checkpoints = [ckpt for ckpt in f]
                checkpoints = [self.extract_global_step(ckpt) for ckpt in checkpoints[1:]]
                checkpoints = list(filter(lambda gs: gs < current_global_step, checkpoints))
                if len(checkpoints) > self.keep_eval_results_max_epoch:
                    checkpoint_to_delete = checkpoints[-self.keep_eval_results_max_epoch]
                    tf.logging.info("Deleting %s results at the step %d", self.mode, checkpoint_to_delete)
                    tfrecord_filespec = os.path.join(self.writer.get_logdir(),
                                                     "eval_result_step{:09d}_*.tfrecord".format(checkpoint_to_delete))
                    alignment_filespec = os.path.join(self.writer.get_logdir(),
                                                      "alignment_eval_result_step{:09d}_*.png".format(
                                                          checkpoint_to_delete))
                    mel_filespec = os.path.join(self.writer.get_logdir(),
                                                "mel_eval_result_step{:09d}_*.png".format(checkpoint_to_delete))
                    for pathname in tf.gfile.Glob([tfrecord_filespec, alignment_filespec, mel_filespec]):
                        file_io.delete_file(pathname)

    def extract_global_step(self, checkpoint_str):
        return int(self.checkpoint_pattern.match(checkpoint_str)[1])


class PostNetMetricsSaver(tf.train.SessionRunHook):

    def __init__(self, global_step_tensor, predicted_spec_tensor, ground_truth_spec_tensor,
                 spec_length_tensor, id_tensor, save_steps,
                 mode, writer: tf.summary.FileWriter):
        self.global_step_tensor = global_step_tensor
        self.predicted_spec_tensor = predicted_spec_tensor
        self.ground_truth_spec_tensor = ground_truth_spec_tensor
        self.spec_length_tensor = spec_length_tensor
        self.id_tensor = id_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, predicted_specs, ground_truth_specs, mel_length, ids = run_context.session.run(
                (self.global_step_tensor, self.predicted_spec_tensor,
                 self.ground_truth_spec_tensor, self.spec_length_tensor, self.id_tensor))
            ids = [str(i) for i in ids]
            id_strings = ",".join(ids)
            result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
            tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)
            write_postnet_training_result(global_step_value, ids, list(predicted_specs),
                                          list(ground_truth_specs), list(mel_length),
                                          filename=os.path.join(self.writer.get_logdir(), result_filename))
            if self.mode == tf.estimator.ModeKeys.EVAL:
                for _id, pred_spec, gt_spec in zip(ids, predicted_specs,
                                                   ground_truth_specs):
                    output_filename = "{}_result_step{:09d}_{}.png".format(self.mode,
                                                                           global_step_value, _id)
                    plot_spec(gt_spec, pred_spec, _id, global_step_value,
                              os.path.join(self.writer.get_logdir(), "spec_" + output_filename))
