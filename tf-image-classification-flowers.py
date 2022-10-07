#! /usr/bin/env python
# git@github.com:jimmygizmo/tensorpic/tf-image-classification-flowers.py
# Version 0.9.0

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


# This program was inspired by the following Tensorflow tutorial. Some text was copied verbatim into the comments.
# https://www.tensorflow.org/tutorials/images/classification

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
SUNFLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
INITIAL_EPOCHS = 10
OPTIMIZED_EPOCHS = 15
TFLITE_MODEL_FILE_PATH = "model-flowers.tflite"

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


log_phase(f"PROJECT:  KERAS IMAGE CLASSIFICATION, ITERATIVE OPTIMIZATION - DEPLOYMENT - FLOWERS DATASET")

log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")
tf_logger_initial_level = tf.get_logger().getEffectiveLevel()
log(f"Tensorflow logger initial effective logging level: {tf_logger_initial_level}")
log(f"INITIAL_EPOCHS: {INITIAL_EPOCHS}")
log(f"OPTIMIZED_EPOCHS: {OPTIMIZED_EPOCHS}")


log_phase(f"PHASE 1:  Download dataset to ~/.keras/datasets/ and count image files. Inspect a sample of images.")

dataset_dir = tf.keras.utils.get_file(
    "flower_photos",
    origin=DATASET_URL,
    untar=True
)

dataset_dir = pathlib.Path(dataset_dir)
log(f"dataset_dir: {dataset_dir}")


image_count = len(list(dataset_dir.glob("*/*.jpg")))  # Nice way to count files.

log(f"image_count: {image_count}")

log(f"dataset_dir listing:\n{pp.pformat(list(dataset_dir.iterdir()))}")

log(f"rose sample - 2 roses:")
roses = list(dataset_dir.glob("roses/*"))
PIL.Image.open(str(roses[0])).show()
PIL.Image.open(str(roses[1])).show()

log(f"tulip sample - 2 tulips:")
tulips = list(dataset_dir.glob("tulips/*"))
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


# TODO: MOST PHASE TITLES NEED RE-WRITING IN THIS PROGRAM AND IN CIFAR - FIX IN ALL PROGRAMS
# TODO: MOST PHASE TITLES NEED RE-WRITING IN THIS PROGRAM AND IN CIFAR - FIX IN ALL PROGRAMS
# TODO: MOST PHASE TITLES NEED RE-WRITING IN THIS PROGRAM AND IN CIFAR - FIX IN ALL PROGRAMS
log_phase(f"PHASE 3:  Examine first 9 images of the dataset. ")

log(f"PLOT:  3x3 - sample of 9 images with labels/categories/classes. ")
log(f"PLOT: * CLOSE PLOT/IMAGE WINDOW TO RESUME EXECUTION *  Execution will pause here on most platforms.")
# TODO: See about making this non-blocking. It is OK to just open the plot/image and then continue without pausing.
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


log(f"Configure the dataset for performance: Adding .cache() and .prefetch(buffer_size=AUTOTUNE)")
AUTOTUNE = tf.data.AUTOTUNE

training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
# Related performance docs: https://www.tensorflow.org/guide/data_performance

log(f"Standardize the data. 0->255 RGB values become 0->1. Small values are better for a neural network.")
normalization_layer = layers.Rescaling(1./255)


# # To use the normalization layer we have two options, do it separately or include it in the model.
# # We want it in the model to assist our deployment. So we will NOT DO the following
# # method of applying it to the data separately
# def data_standardization_manually_upon_dataset_example():
#     normalized_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))
#     image_batch, labels_batch = next(iter(normalized_dataset))
#     first_image = image_batch[0]
#     # Notice the pixel values are now in `[0,1]`.
#     print(np.min(first_image), np.max(first_image))
#
#
# log(f"data_standardization_manually_upon_dataset_example:")
# data_standardization_manually_upon_dataset_example()

log_phase(f"PHASE 4: MODEL - Create basic Keras Sequential model and compile it.")
# The Keras Sequential model consists of three convolution blocks (tf.keras.layers.Conv2D) with a max pooling layer
# (tf.keras.layers.MaxPooling2D) in each of them. There's a fully-connected layer (tf.keras.layers.Dense)
# with 128 units on top of it that is activated by a ReLU activation function ("relu"). This model has not been
# tuned for high accuracy; the goal of this tutorial is to show a standard approach.

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes)
])


log(f"Compile the model: SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']")
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

log(f"Model summary:")
model.summary()


log_phase(f"PHASE 5: TRAINING - Train the model for 10 epochs with Keras model.fit.")


history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS
)

log_phase(f"PHASE 6: VISUALIZE - Assess accuracy over training epochs. One goal is to try to detect overfitting.")


log(f"PLOT: Historical accuracy over epochs comparing accuracy or training vs. validation.")
log(f"PLOT: * CLOSE PLOT/IMAGE WINDOW TO RESUME EXECUTION *  Execution will pause here on most platforms.")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(INITIAL_EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


log_phase(f"PHASE 7: FIRST OPTIMIZATION - Address overfitting using Data Augmentation")

log(f"* Setting tensorflow logger to quieter ERROR level to suppress excessive warnings during data augmentation.")
tf.get_logger().setLevel('ERROR')

log(f"Data Augmentation: RandomFlip, RandomRotation, RandomZoom")
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal",
            input_shape=(img_height, img_width, 3)
        ),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

log(f"Visualize data augmentation: 3x3 of 9 randomly augmented variants of one training image.")

log(f"PLOT: 3x3 - 9 Data Augmentation examples: RandomFlip, RandomRotation, RandomZoom")
log(f"PLOT: * CLOSE PLOT/IMAGE WINDOW TO RESUME EXECUTION *  Execution will pause here on most platforms.")
plt.figure(figsize=(10, 10))
for images, _ in training_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

plt.show()


log_phase(f"PHASE 8: SECOND OPTIMIZATION: - Addressing overfitting. Data augmentation complete, now add Dropout.")

# When you apply dropout to a layer, it randomly drops out (by setting the activation to zero) a number of output
# units from the layer during the training process. Dropout takes a fractional number as its input value, in the
# form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly
# from the applied layer.

log(f"MODEL: Create a new model WITH DROPOUT and train it with the augmented data.")

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, name="outputs")
])

log(f"* Returning tensorflow logger to original level: {tf_logger_initial_level}")
tf.get_logger().setLevel(tf_logger_initial_level)

log(f"COMPILE MODEL: Standard settings, but this is the Dropout model using augmented data.")
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

log(f"Model summary:")
model.summary()


log(f"* Setting tensorflow logger to quieter ERROR level to suppress warnings during training of optimized model.")
tf.get_logger().setLevel('ERROR')

log(f"TRAIN MODEL: Train the Dropout model using the augmented data.")
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=OPTIMIZED_EPOCHS
)

log(f"* Returning tensorflow logger to original level: {tf_logger_initial_level}")
tf.get_logger().setLevel(tf_logger_initial_level)


log_phase(f"PHASE 8: VISUALIZE OPTIMIZED MODEL: - Accuracy of the optimized model with dropout, augmented data.")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(OPTIMIZED_EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label=""'Validation Loss')
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")

log(f"PLOT: Historical accuracy - Optimized model using dropout and augmented data.")
log(f"PLOT: * CLOSE PLOT/IMAGE WINDOW TO RESUME EXECUTION *  Execution will pause here on most platforms.")
plt.show()


log_phase(f"PHASE 9: Predict on new data.")

sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=SUNFLOWER_URL)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

log(f"PREDICTION: This image most likely belongs to {class_names[np.argmax(score)]} "
    f"with a {100 * np.max(score):.2f} percent confidence.")


log_phase(f"PHASE 10: DEPLOYMENT - Convert model to TensorFlow Lite")

log(f"* Setting tensorflow logger to quieter ERROR level to suppress warnings during TensorFlow Lite conversion.")
tf.get_logger().setLevel('ERROR')

log(f"PLOT: Convert the Keras Sequential model to a TensorFlow Lite model.")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

log(f"* Returning tensorflow logger to original level: {tf_logger_initial_level}")
tf.get_logger().setLevel(tf_logger_initial_level)

log(f"DEPLOYABLE MODEL: Writing TensorFlow Lite model to file: model-flowers.tflite")
with open(TFLITE_MODEL_FILE_PATH, "wb") as f:
    f.write(tflite_model)

# The exported model should be about 15 MB.


log_phase(f"PHASE 11: RUN MODEL - Run the TensorFlow Lite model and perform a prediction with it.")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FILE_PATH)

interpreter.get_signature_list()

classify_lite = interpreter.get_signature_runner("serving_default")

# TODO: This bare statement is highly unusual. The purpose is unclear. Could it possibly be a typo in the tutorial?
classify_lite

predictions_lite = classify_lite(sequential_1_input=img_array)["outputs"]
score_lite = tf.nn.softmax(predictions_lite)

assert np.allclose(predictions, predictions_lite)

log(f"TFLITE PREDICTION: This image most likely belongs to {class_names[np.argmax(score_lite)]} "
    f"with a {100 * np.max(score_lite):.2f} percent confidence.")


log_phase(f"PROJECT:  KERAS IMAGE CLASSIFICATION, ITERATIVE OPTIMIZATION DEMONSTRATION COMPLETE.  Exiting.")

