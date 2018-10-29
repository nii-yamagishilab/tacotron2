#!/usr/bin/env python
# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================

from setuptools import setup, find_packages

version = '0.0.1'

setup(name='tacotron2',
      version=version,
      description='',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
      ],
      extras_require={
          "test": [
              "hypothesis",
              "pylint",
          ],
          "train": [
              "pyspark",
              "librosa",
              "docopt",
              "matplotlib",
              "tqdm",
          ],
      }
      )
