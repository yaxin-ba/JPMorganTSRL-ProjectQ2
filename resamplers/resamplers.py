import tensorflow as tf


class MultinomialResample(tf.Module):
    """
    Standard Multinomial Resampling (Classical Bootstrap Particle Filter).
    Draws N independent samples from a categorical distribution based on weights.
    """

    def __init__(self, name="MultinomialResample"):
        super().__init__(name=name)

    @tf.function
    def __call__(self, particles, weights):
        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]

        weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights) / tf.cast(N, tf.float32), weights)
        weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-10)

        indices = tf.random.categorical(tf.math.log(weights + 1e-16), N)
        indices = tf.cast(indices, tf.int32)

        batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, N])
        gather_indices = tf.stack([batch_idx, indices], axis=2)

        resampled_particles = tf.gather_nd(particles, gather_indices)
        uniform_weights = tf.ones_like(weights) / tf.cast(N, tf.float32)

        return resampled_particles, uniform_weights

#
# class SystematicResampler(tf.Module):
#     """Systematic Resampling (Non-differentiable, Low Variance)"""
#
#     def __init__(self, name="SystematicResample"):
#         super().__init__(name=name)
#
#     @tf.function
#     def __call__(self, particles, weights):
#         B = tf.shape(particles)[0]
#         N = tf.shape(particles)[1]
#         N_f = tf.cast(N, tf.float32)
#
#         weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights) / N_f, weights)
#         weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-10)
#
#         # Draw one random offset per batch, then evenly space the rest
#         u = tf.random.uniform([B, 1]) / N_f
#         positions = u + tf.cast(tf.range(N), tf.float32)[tf.newaxis, :] / N_f
#
#         cumulative_sum = tf.cumsum(weights, axis=1)
#         indices = tf.searchsorted(cumulative_sum, positions, side='right')
#         indices = tf.clip_by_value(indices, 0, N - 1)
#         indices = tf.cast(indices, tf.int32)
#
#         batch_indices = tf.tile(tf.expand_dims(tf.range(B), 1), [1, N])
#         gather_indices = tf.stack([batch_indices, indices], axis=-1)
#
#         resampled_particles = tf.gather_nd(particles, gather_indices)
#         uniform_weights = tf.ones_like(weights) / N_f
#
#         return resampled_particles, uniform_weights

class SystematicResampler(tf.Module):
    """Pure-TF Systematic Resampler."""

    @tf.function
    def __call__(self, particles, weights):
        """Allows the resampler to be called directly as an object."""
        return self.resample(particles, weights)

    @tf.function
    def resample(self, particles, weights):
        """Explicit resample method expected by the filter."""
        N = tf.shape(particles)[0]
        N_f = tf.cast(N, tf.float32)

        # Pure TF operations for systematic sampling
        positions = (tf.range(N_f) + tf.random.uniform([])) / N_f
        cumulative_sum = tf.cumsum(weights)

        indices = tf.searchsorted(cumulative_sum, positions, side='right')
        indices = tf.clip_by_value(indices, 0, N - 1)

        new_particles = tf.gather(particles, indices)
        new_weights = tf.ones_like(weights) / N_f

        return new_particles, new_weights