#! /usr/bin/env python
# git@github.com:jimmygizmo/tensorpic/tf-convo-neural-net.py
# Version 1.0.0

print("Initializing Tensorflow.")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stop annoying TF messages about compilation hardware options.

import tensorflow as tf
from keras import datasets
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import pprint


# This code inspired by the following Tensorflow tutorial. A little bit of text was copied verbatim into the comments.
# https://www.tensorflow.org/tutorials/images/cnn

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


log_phase(f"PROJECT:  CONVOLUTIONAL NEURAL NETWORK IMAGE CLASSIFICATION - CIFAR10 DATASET")
log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")


log_phase(f"PHASE 1:  Download dataset. Inspect 25 samples of image and label data.")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, which is why you need the extra index.
    plt.xlabel(class_names[train_labels[i][0]])


log(f"PLOT: Dataset image and label examples.")
plt.show()


log_phase(f"PHASE 2:  Create convolutional base. Use a stack of Conv2D and MaxPooling2D layers.")

model = models.Sequential()
log(f"Object created: model - Type: {type(model)}")

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# The output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels).
# The width and height dimensions tend to shrink as you go deeper in the network.
# The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64).
# Typically, as the width and height shrink, you can afford (computationally) to add more output
# channels in each Conv2D layer.


log(f"Model Summary - Convolutional Base only:")
model.summary()

log(f"Complete model. Add Dense layers on top. Final Dense layer will have 10 outputs.")
# .. feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into one or more Dense layers
# to perform classification. Dense layers take vectors as input (which are 1D), while the current output is
# a 3D tensor. First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top.

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))


log(f"Model Summary - Complete model. Dense Layers added on top (shown at the bottom of the inverted list):")
model.summary()

# The network summary shows that (4, 4, 64) outputs were flattened into vectors of shape (1024)
# before going through two Dense layers.


log_phase(f"COMPILE: Compile the model.")

log(f"MODEL compilation options: losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']")
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


log_phase(f"TRAIN: Train the model.")

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)
log(f"Object created: history - Type: {type(history)}")


log_phase(f"EVALUATE: Evaluate the model.")

log(f"Preparing evaluation plot.")
plt.plot(
    history.history['accuracy'],
    label='accuracy'
)
plt.plot(
    history.history['val_accuracy'],
    label='val_accuracy'
)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

log(f"Evaluating test data.")
test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)

log(f"PLOT: Accuracy history.")
plt.show()

log(f"Evaluation complete. Test accuracy: {test_acc}")


log_phase(f"PROJECT:  CONVOLUTIONAL NEURAL NETWORK IMAGE CLASSIFICATION DEMONSTRATION COMPLETE.  Exiting.")

