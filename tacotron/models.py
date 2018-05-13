import tensorflow as tf
from tacotron.modules import Embedding
from tacotron.tacotron_v1 import EncoderV1, DecoderV1, PostNet
from tacotron.hooks import MetricsSaver
from util.audio import inv_spectrogram_tf


class SingleSpeakerTacotronV1Model(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            encoder = EncoderV1(is_training,
                                cbhg_out_units=params.cbhg_out_units,
                                conv_channels=params.conv_channels,
                                max_filter_width=params.max_filter_width,
                                projection1_out_channels=params.projection1_out_channels,
                                projection2_out_channels=params.projection2_out_channels,
                                num_highway=params.num_highway,
                                prenet_out_units=params.encoder_prenet_out_units,
                                drop_rate=params.drop_rate)

            decoder = DecoderV1(prenet_out_units=params.decoder_prenet_out_units,
                                drop_rate=params.drop_rate,
                                attention_out_units=params.attention_out_units,
                                decoder_out_units=params.decoder_out_units,
                                num_mels=params.num_mels,
                                outputs_per_step=params.outputs_per_step,
                                max_iters=params.max_iters)

            target = labels.mel if (is_training or is_validation) else None

            embedding_output = embedding(features.source)
            encoder_output = encoder(embedding_output)
            mel_output, stop_token, decoder_state = decoder(encoder_output,
                                                is_training=is_training,
                                                is_validation=is_validation,
                                                memory_sequence_length=features.source_length,
                                                target=target)
            alignment = tf.transpose(decoder_state[0].alignment_history.stack(), [1, 2, 0])

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                mel_loss = self.spec_loss(mel_output, labels.mel,
                                          labels.spec_loss_mask)
                done_loss = self.binary_loss(stop_token, labels.done, labels.binary_loss_mask)
                loss = mel_loss + done_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(mel_loss, done_loss, lr)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MetricsSaver([alignment], global_step, mel_output, labels.mel,
                                                   labels.target_length,
                                                   features.id,
                                                   features.text,
                                                   params.alignment_save_steps,
                                                   mode, summary_writer)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=[alignment_saver])

            if is_validation:
                eval_metric_ops = self.get_validation_metrics(mel_loss, done_loss)
                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MetricsSaver([alignment], global_step, mel_output, labels.mel,
                                               labels.target_length,
                                               features.id,
                                               features.text,
                                               1,
                                               mode, summary_writer)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                return tf.estimator.EstimatorSpec(mode, predictions={
                    "id": features.id,
                    "mel": mel_output,
                    "alignment": alignment,
                })

        super(SingleSpeakerTacotronV1Model, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def spec_loss(y_hat, y, mask, n_priority_freq=None, priority_w=0):
        l1_loss = tf.abs(y_hat - y)

        # Priority L1 loss
        if n_priority_freq is not None and priority_w > 0:
            priority_loss = tf.abs(y_hat[:, :, :n_priority_freq] - y[:, :, :n_priority_freq])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        return tf.losses.compute_weighted_loss(l1_loss, weights=tf.expand_dims(mask, axis=2))

    @staticmethod
    def binary_loss(done_hat, done, mask):
        return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1), weights=mask)

    @staticmethod
    def learning_rate_decay(init_rate, global_step):
        warmup_steps = 4000.0
        step = tf.to_float(global_step + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(mel_loss, done_loss, learning_rate):
        if mel_loss is not None:
            tf.summary.scalar("mel_loss", mel_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(mel_loss, done_loss):
        metrics = {}
        if mel_loss is not None:
            metrics["mel_loss"] = tf.metrics.mean(mel_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        return metrics


class TacotronV1PostNetModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_predction = mode == tf.estimator.ModeKeys.PREDICT

            post_net = PostNet(is_training,
                               params.num_freq,
                               params.post_net_cbhg_out_units,
                               params.post_net_conv_channels,
                               params.post_net_max_filter_width,
                               params.post_net_projection1_out_channels,
                               params.post_net_projection2_out_channels,
                               params.post_net_num_highway)

            linear_output = post_net(features.mel)

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                linear_loss = self.spec_loss(linear_output, labels.spec,
                                             labels.spec_loss_mask)
                loss = linear_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(linear_loss, lr)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            if is_validation:
                eval_metric_ops = self.get_validation_metrics(linear_loss)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  eval_metric_ops=eval_metric_ops)

            if is_predction:
                return tf.estimator.EstimatorSpec(mode, predictions={
                    "id": features.id,
                    "spec": linear_output,
                    "audio": inv_spectrogram_tf(linear_output)
                })

        super(TacotronV1PostNetModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def spec_loss(y_hat, y, mask, n_priority_freq=None, priority_w=0):
        l1_loss = tf.abs(y_hat - y)

        # Priority L1 loss
        if n_priority_freq is not None and priority_w > 0:
            priority_loss = tf.abs(y_hat[:, :, :n_priority_freq] - y[:, :, :n_priority_freq])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        return tf.losses.compute_weighted_loss(l1_loss, weights=tf.expand_dims(mask, axis=2))

    @staticmethod
    def learning_rate_decay(init_rate, global_step):
        warmup_steps = 4000.0
        step = tf.to_float(global_step + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(linear_loss, learning_rate):
        if linear_loss is not None:
            tf.summary.scalar("linear_loss", linear_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(linear_loss):
        metrics = {}
        if linear_loss is not None:
            metrics["linear_loss"] = tf.metrics.mean(linear_loss)
        return metrics
