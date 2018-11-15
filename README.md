# Tacotron2

This is an implementation of Tacotron and Tacotron2.


## Requirements

- Python >= 3.6
- tensorflow >= 1.11
- pyspark >= 2.3.0
- librosa >= 0.6.1
- scipy >= 1.1.1
- matplotlib >= 2.2.2
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


## ToDo

- [ ] Add Tacotron2 model
- [ ] Implement L2 regularization
- [ ] More easy to use runnable example

## Authors

- Yusuke Yasuda (National Institute of Informatics, Japan) @TanUkkii007


## References and resources

This is an implementation of the following papers.

- Yuxuan  Wang,  R.J.  Skerry-Ryan,  Daisy  Stanton,  Yonghui
Wu,  Ron  J.  Weiss,  Navdeep  Jaitly,  Zongheng  Yang,  Ying
Xiao,   Zhifeng   Chen,   Samy   Bengio,   Quoc   Le,   Yannis
Agiomyrgiannakis, Rob Clark, and Rif A. Saurous, “Tacotron:
Towards end-to-end speech synthesis,”   in
Proc. Interspeech
,
2017, pp. 4006–4010
- Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster,
Navdeep  Jaitly,  Zongheng  Yang,  Zhifeng  Chen,  Yu  Zhang,
Yuxuan  Wang,   RJ-Skerrv  Ryan,   Rif  A.  Saurous,   Yannis
Agiomyrgiannakis, and Yonghui Wu,  “Natural TTS synthesis
by conditioning WaveNet on Mel spectrogram predictions,” in
Proc. ICASSP
, 2018, pp. 4779–4783

This implementation is inspired from the following pioneers.
- https://github.com/keithito/tacotron
- https://github.com/Rayhane-mamah/Tacotron-2

Thank for outstanding papers and implementations.

## Licence

BSD 3-Clause License

Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
