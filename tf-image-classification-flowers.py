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


log_phase(f"PHASE 2:  Create datasets. ")


batch_size = 32
img_height = 180
img_width = 180

log(f"training_dataset: image_dataset_from_directory - validation_split=0.2: {dataset_dir}")
training_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


log(f"validation_dataset: image_dataset_from_directory - validation_split=0.2: {dataset_dir}")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = training_dataset.class_names
log(f"Class names (class_names):\n{class_names}")


log_phase(f"PHASE 3:  Examine first 9 images of the dataset. ")

log(f"PLOT:  3x3 - sample of 9 images with labels/categories/classes. ")
plt.figure(figsize=(10, 10))
for images, labels in training_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()


# Below is text from the tutorial. TODO: Clarify the "retrieve batches" aspect.
# The little disabled block of code returns a few tuples and stops.
# You will pass these datasets to the Keras Model.fit method for training later in this tutorial.
# If you like, you can also manually iterate over the dataset and retrieve batches of images:
# for image_batch, labels_batch in training_dataset:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

AUTOTUNE = tf.data.AUTOTUNE

training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

