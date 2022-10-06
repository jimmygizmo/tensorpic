#! /usr/bin/env python
# git@github.com:jimmygizmo/tensorpic/tf-image-classification-flowers.py
# Version 0.5.0

print("Initializing Tensorflow.")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Stop annoying TF messages about compilation hardware options.

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pprint
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image


# This code inspired by the following Tensorflow tutorial. A little bit of text was copied verbatim into the comments.
# https://www.tensorflow.org/tutorials/images/classification

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


log_phase(f"PROJECT: IMAGE CLASSIFICATION, ITERATIVE TRAINING OPTIMIZATION - DEPLOYMENT - FLOWERS DATASET")
log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")


log_phase(f"PHASE 1:  Download dataset to ~/.keras/datasets/ and count image files. Inspect a sample of images.")

dataset_dir = tf.keras.utils.get_file(
    'flower_photos',
    origin=DATASET_URL,
    untar=True
)

dataset_dir = pathlib.Path(dataset_dir)
log(f"dataset_dir: {dataset_dir}")


image_count = len(list(dataset_dir.glob("*/*.jpg")))  # Nice way to count files.

log(f"image_count: {image_count}")

log(f"dataset_dir listing:\n{pp.pformat(list(dataset_dir.iterdir()))}")

log(f"rose sample - 2 roses:")
roses = list(dataset_dir.glob('roses/*'))
PIL.Image.open(str(roses[0])).show()
PIL.Image.open(str(roses[1])).show()

log(f"tulip sample - 2 tulips:")
tulips = list(dataset_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0])).show()
PIL.Image.open(str(tulips[1])).show()



