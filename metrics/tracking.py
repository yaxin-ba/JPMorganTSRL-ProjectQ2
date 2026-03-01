import tensorflow as tf
import time


class PerformanceMetrices(tf.Module):
    """Unified mathematical evaluation for Particle Filters."""

    @staticmethod
    @tf.function
    def calculate_rmse(estimate, truth):
        mse = tf.reduce_mean(tf.square(estimate - truth))
        return tf.sqrt(mse)

    @staticmethod
    def calculate_grad_norm(tape, loss, inputs):
        grads = tape.gradient(loss, inputs)
        if grads is None:
            return tf.constant(0.0)
        return tf.norm(grads)


class SystemProfiler:
    """Tracks computational efficiency and memory footprint."""

    @staticmethod
    def get_model_size(model):
        """Calculates storage footprint (Number of Trainable Parameters)."""
        # [Image of neural network parameter count calculation]
        total_params = 0
        for var in model.trainable_variables:
            total_params += tf.reduce_prod(var.shape)
        return total_params

    @staticmethod
    def measure_latency(model, obs_batch):
        """Measures execution speed (Latency) in milliseconds."""
        # Warmup
        _ = model(obs_batch)

        start = time.perf_counter()
        _ = model(obs_batch)
        end = time.perf_counter()

        return (end - start) * 1000  # Convert to ms