"""
Contains the networks which describe each agent.
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.losses import mean_squared_error
import pickle
from numba import cuda
import gc

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from copy import copy
from time import time
import random

# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
