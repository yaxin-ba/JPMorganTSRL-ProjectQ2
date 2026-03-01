import tensorflow as tf


class SoftResample(tf.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    @tf.function
    def __call__(self, particles, weights):
        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]
        N_float = tf.cast(N, tf.float32)

        weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights) / N_float, weights)
        weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-10)

        uniform = tf.ones_like(weights) / N_float
        soft_weights = self.alpha * weights + (1.0 - self.alpha) * uniform
        soft_weights = tf.maximum(soft_weights, 0.0)
        soft_weights = soft_weights / (tf.reduce_sum(soft_weights, axis=1, keepdims=True) + 1e-10)

        logits = tf.math.log(soft_weights + 1e-16)
        indices = tf.random.categorical(logits, num_samples=N, dtype=tf.int32)

        batch_indices = tf.tile(tf.expand_dims(tf.range(B), 1), [1, N])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        new_particles = tf.gather_nd(particles, gather_indices)

        w_selected = tf.gather_nd(weights, gather_indices)
        soft_selected = tf.gather_nd(soft_weights, gather_indices)

        new_weights = w_selected / (soft_selected + 1e-10)
        new_weights = new_weights / (tf.reduce_sum(new_weights, axis=1, keepdims=True) + 1e-10)

        return new_particles, new_weights

    @tf.function
    def resample(self, particles, weights):
        return self.__call__(particles, weights)


class OTResample(tf.Module):
    def __init__(self, epsilon=0.1, n_iters=10):
        super().__init__()
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        # Store as TF Constant to prevent retracing warnings
        self.n_iters = tf.constant(n_iters, dtype=tf.int32)

    @tf.function
    def __call__(self, particles, weights):
        N_float = tf.cast(tf.shape(particles)[1], tf.float32)

        weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights) / N_float, weights)
        if len(weights.shape) > 2: weights = tf.squeeze(weights, -1)

        x_i = tf.expand_dims(particles, 2)
        x_j = tf.expand_dims(particles, 1)
        C = tf.reduce_sum(tf.square(x_i - x_j), axis=-1)

        C_mean = tf.stop_gradient(tf.reduce_mean(C, axis=[1, 2], keepdims=True) + 1e-8)
        C_scaled = C / C_mean

        log_mu = tf.math.log(weights + 1e-16)
        log_nu = tf.math.log(tf.ones_like(weights) / N_float)
        log_K = -C_scaled / self.epsilon

        f = tf.zeros_like(weights)
        g = tf.zeros_like(weights)

        for _ in tf.range(self.n_iters):
            term_g = tf.expand_dims(g, 1) + log_K
            f = log_mu - tf.reduce_logsumexp(term_g, axis=2)
            term_f = tf.expand_dims(f, 2) + log_K
            g = log_nu - tf.reduce_logsumexp(term_f, axis=1)

        log_P = tf.expand_dims(f, 2) + tf.expand_dims(g, 1) + log_K
        P = tf.exp(log_P)

        new_particles = N_float * tf.matmul(P, particles, transpose_a=True)
        new_weights = tf.ones_like(weights) / N_float

        return new_particles, new_weights

    @tf.function
    def resample(self, particles, weights):
        return self.__call__(particles, weights)
