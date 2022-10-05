#! /usr/bin/env python
# git@github.com:jimmygizmo/tensorpic/tf-convo-neural-net.py
# Version 0.9.0

import tensorflow as tf
from keras import datasets
from keras import layers
from keras import models
import matplotlib.pyplot as plt


# https://www.tensorflow.org/tutorials/images/cnn

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0









