import tensorflow as tf

hparams = tf.contrib.training.HParams(

    # Audio
    num_mels=80,
    num_freq=1025,
    sample_rate=20000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    min_level_db=-100,
    ref_level_db=20,
)