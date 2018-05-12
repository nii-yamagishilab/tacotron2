import tensorflow as tf
from collections import namedtuple
from abc import abstractmethod
from util.tfrecord import decode_preprocessed_source_data, decode_preprocessed_target_data, \
    parse_preprocessed_source_data, parse_preprocessed_target_data, \
    PreprocessedSourceData, PreprocessedTargetData


class SourceData(namedtuple("SourceData",
                            ["id", "text", "source", "source_length",
                             "text2", "source2", "source_length2"])):
    pass


class TargetData(
    namedtuple("TargetData",
               ["id", "spec", "spec_width", "mel", "mel_width", "target_length", "done",
                "spec_loss_mask", "binary_loss_mask"])):
    pass


class PostNetSourceData(
    namedtuple("PostNetSourceData",
               ["id", "mel", "mel_width", "target_length"])):
    pass


class PostNetTargetData(
    namedtuple("PostNetTargetData",
               ["id", "spec", "spec_width", "target_length", "spec_loss_mask"])):
    pass


class SourceDataWithMelPrediction(
    namedtuple("SourceDataWithMelPrediction",
               ["id",
                "spec", "spec_width",
                "ground_truth_mel", "ground_truth_mel_width", "ground_truth_target_length",
                "mel", "mel_width", "target_length"])):
    pass


class PredictedMel(
    namedtuple("PredictedMel",
               ["id", "predicted_mel", "predicted_mel_width", "predicted_target_length", "alignment"])):
    pass


class DatasetSource:

    def __init__(self, source, target, hparams):
        self._source = source
        self._target = target
        self._hparams = hparams

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def hparams(self):
        return self._hparams

    def prepare_and_zip(self):
        zipped = tf.data.Dataset.zip((self._prepare_source(), self._prepare_target()))
        return ZippedDataset(zipped, self.hparams)

    def _prepare_source(self):
        def convert(inputs: PreprocessedSourceData):
            return SourceData(inputs.id, inputs.text, inputs.source, inputs.source_length,
                              inputs.text2, inputs.source2, inputs.source_length2)

        return self._decode_source().map(lambda inputs: convert(inputs))

    def _prepare_target(self):
        def convert(target: PreprocessedTargetData):
            r = self.hparams.outputs_per_step

            target_length = target.target_length
            padded_target_length = (target_length // r + 1) * r

            # spec and mel length must be multiple of outputs_per_step
            def padding_function(t):
                tail_padding = padded_target_length - target_length
                padding_shape = tf.sparse_tensor_to_dense(
                    tf.SparseTensor(indices=[(0, 1)], values=tf.expand_dims(tail_padding, axis=0), dense_shape=(2, 2)))
                return lambda: tf.pad(t, paddings=padding_shape)

            no_padding_condition = tf.equal(tf.to_int64(0), target_length % r)

            spec = tf.cond(no_padding_condition, lambda: target.spec, padding_function(target.spec))
            mel = tf.cond(no_padding_condition, lambda: target.mel, padding_function(target.mel))

            spec.set_shape((None, self.hparams.num_freq))
            mel.set_shape((None, self.hparams.num_mels))

            padded_target_length = tf.cond(no_padding_condition, lambda: target_length, lambda: padded_target_length)

            # done flag
            done = tf.concat([tf.zeros(padded_target_length // r - 1, dtype=tf.float32),
                              tf.ones(1, dtype=tf.float32)], axis=0)

            # loss mask
            spec_loss_mask = tf.ones(shape=padded_target_length, dtype=tf.float32)
            binary_loss_mask = tf.ones(shape=padded_target_length // r, dtype=tf.float32)

            return TargetData(target.id, spec, target.spec_width, mel, target.mel_width, padded_target_length, done,
                              spec_loss_mask, binary_loss_mask)

        return self._decode_target().map(lambda inputs: convert(inputs))

    def _decode_source(self):
        return self.source.map(lambda d: decode_preprocessed_source_data(parse_preprocessed_source_data(d)))

    def _decode_target(self):
        return self.target.map(lambda d: decode_preprocessed_target_data(parse_preprocessed_target_data(d)))


class DatasetBase:

    @abstractmethod
    def apply(self, dataset, hparams):
        raise NotImplementedError("apply")

    @property
    @abstractmethod
    def dataset(self):
        raise NotImplementedError("dataset")

    @property
    @abstractmethod
    def hparams(self):
        raise NotImplementedError("hparams")

    def filter(self, predicate):
        return self.apply(self.dataset.filter(predicate), self.hparams)

    def filter_by_max_output_length(self):
        def predicate(s, t: PreprocessedTargetData):
            max_output_length = self.hparams.max_iters * self.hparams.outputs_per_step
            return tf.less_equal(t.target_length, max_output_length)

        return self.filter(predicate)

    def shuffle(self, buffer_size):
        return self.apply(self.dataset.shuffle(buffer_size), self.hparams)

    def repeat(self, count=None):
        return self.apply(self.dataset.repeat(count), self.hparams)


class ZippedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return ZippedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def group_by_batch(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.hparams.batch_size
        approx_min_target_length = self.hparams.approx_min_target_length
        bucket_width = self.hparams.batch_bucket_width
        num_buckets = self.hparams.batch_num_buckets

        def key_func(source, target):
            target_length = tf.minimum(target.target_length - approx_min_target_length, 0)
            bucket_id = target_length // bucket_width
            return tf.minimum(tf.to_int64(num_buckets), bucket_id)

        def reduce_func(unused_key, window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=(
                SourceData(
                    id=tf.TensorShape([]),
                    text=tf.TensorShape([]),
                    source=tf.TensorShape([None]),
                    source_length=tf.TensorShape([]),
                    text2=tf.TensorShape([]),
                    source2=tf.TensorShape([None]),
                    source_length2=tf.TensorShape([]),
                ),
                TargetData(
                    id=tf.TensorShape([]),
                    spec=tf.TensorShape([None, self.hparams.num_freq]),
                    spec_width=tf.TensorShape([]),
                    mel=tf.TensorShape([None, self.hparams.num_mels]),
                    mel_width=tf.TensorShape([]),
                    target_length=tf.TensorShape([]),
                    done=tf.TensorShape([None]),
                    spec_loss_mask=tf.TensorShape([None]),
                    binary_loss_mask=tf.TensorShape([None]),
                )), padding_values=(
                SourceData(
                    id=tf.to_int64(0),
                    text="",
                    source=tf.to_int64(0),
                    source_length=tf.to_int64(0),
                    text2="",
                    source2=tf.to_int64(0),
                    source_length2=tf.to_int64(0),
                ),
                TargetData(
                    id=tf.to_int64(0),
                    spec=tf.to_float(0),
                    spec_width=tf.to_int64(0),
                    mel=tf.to_float(0),
                    mel_width=tf.to_int64(0),
                    target_length=tf.to_int64(0),
                    done=tf.to_float(1),
                    spec_loss_mask=tf.to_float(0),
                    binary_loss_mask=tf.to_float(0),
                )))

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size * 5))
        return BatchedDataset(batched, self.hparams)


class BatchedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return BatchedDataset(self.dataset, self.hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams


class PostNetDatasetSource:

    def __init__(self, target, hparams):
        self._target = target
        self._hparams = hparams

    @property
    def hparams(self):
        return self._hparams

    def create_source_and_target(self):
        return PostNetPairedDataset(self._prepare_target(), self.hparams)

    def _prepare_target(self):
        def convert(target: PreprocessedTargetData):
            r = self.hparams.outputs_per_step

            target_length = target.target_length
            padded_target_length = (target_length // r + 1) * r

            # spec and mel length must be multiple of outputs_per_step
            def padding_function(t):
                tail_padding = padded_target_length - target_length
                padding_shape = tf.sparse_tensor_to_dense(
                    tf.SparseTensor(indices=[(0, 1)], values=tf.expand_dims(tail_padding, axis=0), dense_shape=(2, 2)))
                return lambda: tf.pad(t, paddings=padding_shape)

            no_padding_condition = tf.equal(tf.to_int64(0), target_length % r)

            spec = tf.cond(no_padding_condition, lambda: target.spec, padding_function(target.spec))
            mel = tf.cond(no_padding_condition, lambda: target.mel, padding_function(target.mel))

            spec.set_shape((None, self.hparams.num_freq))
            mel.set_shape((None, self.hparams.num_mels))

            padded_target_length = tf.cond(no_padding_condition, lambda: target_length, lambda: padded_target_length)

            # loss mask
            spec_loss_mask = tf.ones(shape=padded_target_length, dtype=tf.float32)

            return (PostNetSourceData(target.id, mel, target.mel_width, padded_target_length),
                    PostNetTargetData(target.id, spec, target.spec_width, padded_target_length, spec_loss_mask))

        return self._decode_target().map(lambda inputs: convert(inputs))

    def _decode_target(self):
        return self._target.map(lambda d: decode_preprocessed_target_data(parse_preprocessed_target_data(d)))


class PostNetPairedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return PostNetPairedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def group_by_batch(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.hparams.batch_size
        approx_min_target_length = self.hparams.approx_min_target_length
        bucket_width = self.hparams.batch_bucket_width
        num_buckets = self.hparams.batch_num_buckets

        def key_func(source, target):
            target_length = tf.minimum(source.target_length - approx_min_target_length, 0)
            bucket_id = target_length // bucket_width
            return tf.minimum(tf.to_int64(num_buckets), bucket_id)

        def reduce_func(unused_key, window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=(
                PostNetSourceData(
                    id=tf.TensorShape([]),
                    mel=tf.TensorShape([None, self.hparams.num_mels]),
                    mel_width=tf.TensorShape([]),
                    target_length=tf.TensorShape([]),
                ),
                PostNetTargetData(
                    id=tf.TensorShape([]),
                    spec=tf.TensorShape([None, self.hparams.num_freq]),
                    spec_width=tf.TensorShape([]),
                    target_length=tf.TensorShape([]),
                    spec_loss_mask=tf.TensorShape([None]),
                )), padding_values=(
                PostNetSourceData(
                    id=tf.to_int64(0),
                    mel=tf.to_float(0),
                    mel_width=tf.to_int64(0),
                    target_length=tf.to_int64(0),
                ),
                PostNetTargetData(
                    id=tf.to_int64(0),
                    spec=tf.to_float(0),
                    spec_width=tf.to_int64(0),
                    target_length=tf.to_int64(0),
                    spec_loss_mask=tf.to_float(0),
                )))

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size * 5))
        return BatchedDataset(batched, self.hparams)

    def combine_with_prediction(self, predicted_mel_dataset):
        def combine_func(source_and_target, prediction: PredictedMel):
            source, target = source_and_target
            source_with_prediction = SourceDataWithMelPrediction(
                id=source.id,
                spec=target.spec,
                spec_width=target.spec_width,
                ground_truth_mel=source.mel,
                ground_truth_mel_width=source.mel_width,
                ground_truth_target_length=source.target_length,
                mel=prediction.predicted_mel,
                mel_width=prediction.predicted_mel_width,
                target_length=prediction.predicted_target_length,
            )
            return source_with_prediction

        dataset = tf.data.Dataset.zip((self.dataset, predicted_mel_dataset)).map(combine_func)
        return PredictionDataset(dataset, self.hparams)


class PredictionDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return PredictionDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def expand_batch_dim(self):
        batch_size = 1

        def key_func(source):
            return source.id

        def reduce_func(unused_key, window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=SourceDataWithMelPrediction(
                id=tf.TensorShape([]),
                spec=tf.TensorShape([None, self.hparams.num_freq]),
                spec_width=tf.TensorShape([]),
                ground_truth_mel=tf.TensorShape([None, self.hparams.num_mels]),
                ground_truth_mel_width=tf.TensorShape([]),
                ground_truth_target_length=tf.TensorShape([]),
                mel=tf.TensorShape([None, self.hparams.num_mels]),
                mel_width=tf.TensorShape([]),
                target_length=tf.TensorShape([]),
            ), padding_values=SourceDataWithMelPrediction(
                id=tf.to_int64(0),
                spec=tf.to_float(0),
                spec_width=tf.to_int64(0),
                ground_truth_mel=tf.to_float(0),
                ground_truth_mel_width=tf.to_int64(0),
                ground_truth_target_length=tf.to_int64(0),
                mel=tf.to_float(0),
                mel_width=tf.to_int64(0),
                target_length=tf.to_int64(0),
            ))

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size * 5))
        paired = batched.map(lambda v: (v, v))
        return BatchedDataset(paired, self.hparams)
