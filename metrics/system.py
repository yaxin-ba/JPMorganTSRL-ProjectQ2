import tensorflow as tf
import time


def measure_latency(model, inputs):
    """Measures execution time in milliseconds."""
    # Warmup to exclude graph compilation time
    _ = model(inputs)

    start = time.perf_counter()
    _ = model(inputs)
    return (time.perf_counter() - start) * 1000


def get_parameter_count(model):
    """Returns the number of trainable variables (storage size)."""
    return sum([tf.reduce_prod(v.shape) for v in model.trainable_variables])