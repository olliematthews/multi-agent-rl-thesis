import tensorflow as tf

from .actor import Actor
from .critic import Critic
from .state_normaliser import StateNormaliser

__all__ = ["Actor", "Critic", "StateNormaliser"]

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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
