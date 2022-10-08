#! /usr/bin/env python
# git@github.com:jimmygizmo/tensorpic/tf-image-classification-flowers.py
# Version 0.5.0

print("Initializing Tensorflow.")
import tensorflow as tf
import pprint


# This program was inspired by the following Tensorflow tutorial. Some text was copied verbatim into the comments.
# https://www.tensorflow.org/guide/gpu

# Related guide: Optimize TensorFlow GPU Performance
# https://www.tensorflow.org/guide/gpu_performance_analysis

CONSTANT = "blah"

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


log_phase(f"PROJECT:  GPU USAGE - DISTRIBUTION STRATEGIES - FINE-GRAINED CONTROL")

log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")
tf_logger_initial_level = tf.get_logger().getEffectiveLevel()
log(f"Tensorflow logger initial effective logging level: {tf_logger_initial_level}")

available_gpus = tf.config.list_physical_devices('GPU')
available_gpu_count = len(available_gpus)
log(f"Number of available GPUs: {available_gpu_count}")
log(f"Available GPUs: {available_gpus}")

# "/device:CPU:0": The CPU of your machine.
# "/GPU:0": Short-hand notation for the first GPU of your machine that is visible to TensorFlow.
# "/job:localhost/replica:0/task:0/device:GPU:1": Fully qualified name of the second GPU of
#     your machine that is visible to TensorFlow.

log(f"Turning on device placement logging so we can see GPU/CPU assignment. (tf.debugging)")
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

# The following code will force certain operations on the CPU, whereas they would have otherwise
# defaulted to the GPU. The MatMul should default to any available GPU.

log(f"Tensors forced onto CPU. MatMul operation will run on GPU if possible.")
tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Tensors will be automatically copied between devices if required.

# Run on the GPU
c = tf.matmul(a, b)
print(c)


# By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible
# to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices
# by reducing memory fragmentation. To limit TensorFlow to a specific set of GPUs, use the tf.config.set
# visible_devices method.
log(f"Restrict TensorFlow to only use the first GPU.")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        log(f"Physical GPU count: {len(gpus)}    Logical GPU count: {len(logical_gpus)}")
    except RuntimeError as e:
        log(f"*** EXCEPTION ***: RuntimeError")
        # Visible devices must be set before GPUs have been initialized
        print(e)


# In some cases it is desirable for the process to only allocate a subset of the available memory, or to only grow
# the memory usage as is needed by the process. TensorFlow provides two methods to control this.

# The first option is to turn on memory growth by calling tf.config.experimental.set_memory_growth, which attempts
# to allocate only as much GPU memory as needed for the runtime allocations: it starts out allocating very little
# memory, and as the program gets run and more GPU memory is needed, the GPU memory region is extended for the
# TensorFlow process. Memory is not released since it can lead to memory fragmentation. To turn on memory growth
# for a specific GPU, use the following code prior to allocating any tensors or executing any ops.

log(f"Limiting GPU memory growth: Setting experimental memory growth control to True on GPUs.")
# TODO: Clarify, does it touch all GPUs or only those "visible" as per above. Look closer at list_physical_devices.
#   Need a multi-GPU environment to test this.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            log(f"Setting experimental memory growth control to True on GPU: {gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            log(f"Physical GPU count: {len(gpus)}    Logical GPU count: {len(logical_gpus)}")
    except RuntimeError as e:
        log(f"*** EXCEPTION ***: RuntimeError")
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Another way to enable this option is to set the environmental variable TF_FORCE_GPU_ALLOW_GROWTH to true.
# This configuration is platform specific.

# The second method is to configure a virtual GPU device with tf.config.set_logical_device_configuration and set
# a hard limit on the total memory to allocate on the GPU.

log(f"Configure a virtual GPU device. Set a hard limit on the total memory to allocate to the GPU.")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        log(f"Physical GPU count: {len(gpus)}    Logical GPU count: {len(logical_gpus)}")
    except RuntimeError as e:
        log(f"*** EXCEPTION ***: RuntimeError")
        # Virtual devices must be set before GPUs have been initialized
        print(e)


