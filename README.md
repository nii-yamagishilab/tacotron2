# Tacotron2


## Requirements

- Python >= 3.6
- tensorflow >= 1.11
- pyspark >= 2.3.0
- librosa >= 0.6.1
- scipy >= 1.1.1
- hypothesis >= 3.59.1


## Preprocessing

```bash
preprocess.py <dataset> </input/dataset/dir> </output/dataset/dir>
```

Currently available dataset are,

- ljspeech
- blizzard2012


## Training

For training Tacotron itself, run the following command.

```bash
train.py --dataset=<dataset> --data-root=</output/dataset/dir> --checkpoint-dir=</path/to/model/dir> --hparams=<parmas>
```

For training Post-net of Tacotron (Mel to linear spectrogram conversion), run the following command.

```bash
train_postnet.py --dataset=<dataset> --data-root=</output/dataset/dir> --checkpoint-dir=</path/to/postnet/model/dir> --hparams=<parmas>
```

See [Preprocessing](#Preprocessing) for available dataset.



## Synthesis

```bash
synthesize.py  --dataset=<dataset> --data-root=</output/dataset/dir> --checkpoint-dir=</path/to/model/dir> --postnet-checkpoint-dir=</path/to/postnet/model/dir> --hparams=<parmas>

```



## How to use as an external library

This implementation supports Bazel build. You can add this repository as a external dependency in your Bazel project.

Add following lines to a `WORKSPACE` file of your project.
These lines configure how to get Tacotron2 codes and what version you use.

```
git_repository(
    name = "tacotron2",
    remote = "git@github.com:nii-yamagishilab/tacotron2.git",
    commit = "138c7934e3c6d99238f8b6b84d6b0a30f4ea8b2e",
)
```

Then add a dependency of Tacotron2 to your `BUILD` file.
For example, adding following lines enables your training script to use Tacotron2 codes.

```
py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        "@tacotron2//:tacotron2",
    ],
)
```

Now you can import `tacotron2` package in your training script.