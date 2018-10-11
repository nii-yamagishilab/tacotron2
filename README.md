# Tacotron2





# How to use as an external library

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