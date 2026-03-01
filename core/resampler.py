import tensorflow as tf


class BaseResampler(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function
    def resample(self, particles, log_weights):
        raise NotImplementedError