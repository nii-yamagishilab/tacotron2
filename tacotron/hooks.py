import tensorflow as tf
import os
import numpy as np
from typing import List
from util.tfrecord import write_tfrecord, int64_feature, bytes_feature
from util.metrics import plot_alignment, plot_mel


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


class MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, alignment_tensors, global_step_tensor, predicted_mel_tensor, ground_truth_mel_tensor,
                 mel_length_tensor, id_tensor,
                 text_tensor, save_steps,
                 mode, writer: tf.summary.FileWriter):
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
            id_strings = ",".join([str(i) for i in ids])
            result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
            tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)
            write_training_result(global_step_value, list(ids), list(texts), list(predicted_mels),
                                  list(ground_truth_mels), list(mel_length),
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
