#! /usr/bin/env python
# git@github.com:jimmygizmo/tensorpic/tf-performance-tune-bench.py
# Version 0.5.0

print("Initializing Tensorflow.")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Stop annoying TF messages about compilation hardware options.

import tensorflow as tf
import pprint


# This program was inspired by the following Tensorflow tutorial. Some text was copied verbatim into the comments.
# https://www.tensorflow.org/guide/data_performance

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


log_phase(f"PROJECT:  HIGH PERFORMANCE INPUT PIPELINES - TF.DATA API")
log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")


# TODO: This program stub is a planned project, set aside temporarily while other projects get attention. Oct. 2022.

