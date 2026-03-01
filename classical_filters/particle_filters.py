import tensorflow as tf

# Fallback import for standalone testing
try:
    from core.filter import BaseFilter
except ImportError:
    class BaseFilter(tf.Module):
        def __init__(self, ssm=None, name=None):
            super().__init__(name=name)
            self.ssm = ssm


class ParticleFilter(BaseFilter):
    def __init__(self, resampler, ssm=None, n_particles=100, ess_threshold=0.5, name="ParticleFilter"):
        super().__init__(ssm, name=name)
        self.resampler = resampler
        self.N = tf.constant(n_particles, dtype=tf.int32)
        self.N_f = tf.cast(n_particles, tf.float32)
        self.ess_threshold = tf.constant(ess_threshold, dtype=tf.float32)

    @tf.function
    def predict(self, particles, t=None):
        particles_pred = self.ssm.f_fn(particles, t)
        noise = tf.random.normal(tf.shape(particles_pred), stddev=tf.sqrt(self.ssm.Q[0, 0]))
        return particles_pred + noise

    @tf.function
    def update(self, particles_pred, weights_prev, y, t=None):
        # 1. Predict Observations
        pred_y = self.ssm.h_fn(particles_pred, t)

        # 2. General Log Likelihood (handles batches and multi-dimensional obs)
        y_expanded = tf.expand_dims(y, axis=-2)
        diff = pred_y - y_expanded

        R_inv = tf.linalg.inv(self.ssm.R)
        mapped_diff = tf.linalg.matmul(diff, R_inv)
        dist = tf.reduce_sum(mapped_diff * diff, axis=-1)

        log_liks = -0.5 * dist
        log_weights = tf.math.log(weights_prev + 1e-300) + log_liks

        # 3. Normalize
        log_weights_max = tf.reduce_max(log_weights, axis=-1, keepdims=True)
        w_unnorm = tf.exp(log_weights - log_weights_max)
        weights_update = w_unnorm / tf.reduce_sum(w_unnorm, axis=-1, keepdims=True)

        x_est = tf.reduce_sum(tf.expand_dims(weights_update, -1) * particles_pred, axis=-2)
        ess = 1.0 / tf.reduce_sum(tf.square(weights_update), axis=-1)

        return particles_pred, weights_update, x_est, ess

    @tf.function
    def resample_if_needed(self, particles, weights, ess):
        def do_resample():
            return self.resampler.resample(particles, weights)

        def skip_resample():
            return particles, weights

        mean_ess = tf.reduce_mean(ess)

        particles_resampled, weights_resampled = tf.cond(
            mean_ess < (self.ess_threshold * self.N_f),
            true_fn=do_resample,
            false_fn=skip_resample
        )

        return particles_resampled, weights_resampled

#
# class DifferentiableParticleFilter(BaseFilter):
#     """
#     Generic Differentiable Particle Filter.
#     Connects any valid SSM with a DPF Resampler to maintain end-to-end auto-gradients.
#     """
#
#     def __init__(self, ssm, resampler, n_particles=100, name="DPF"):
#         super().__init__(ssm=ssm, name=name)
#         self.resampler = resampler
#         self.N = tf.constant(n_particles, dtype=tf.int32)
#
#     @tf.function
#     def __call__(self, observations):
#         B = tf.shape(observations)[0]
#         T = tf.shape(observations)[1]
#         D = tf.shape(observations)[2]
#
#         particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
#         weights = tf.ones([B, self.N]) / tf.cast(self.N, tf.float32)
#
#         est_states = tf.TensorArray(tf.float32, size=T)
#         ess_history = tf.TensorArray(tf.float32, size=T)
#
#         for t in tf.range(T):
#             obs = observations[:, t]
#             t_f = tf.cast(t + 1, tf.float32)
#
#             if t > 0 and self.resampler is not None:
#                 # particles, weights = self.resampler.resample(particles, weights)
#                 particles, weights = self.resampler(particles, weights)
#
#             # Predict
#             noise = tf.random.normal(tf.shape(particles)) * tf.sqrt(self.ssm.Q[0, 0])
#             particles = self.ssm.f_fn(particles, t_f) + noise
#
#             # Update Likelihood
#             pred_obs = self.ssm.h_fn(particles, t_f)
#             dist = tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs, 1)), axis=2)
#             log_lik = -0.5 * dist / self.ssm.R[0, 0]
#
#             # Bound the minimum probability mass to prevent zero-weight collapse
#             lik = tf.exp(log_lik)
#             weights = weights * lik + 1e-8
#             weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
#
#             est = tf.reduce_sum(tf.expand_dims(weights, 2) * particles, axis=1)
#             est_states = est_states.write(t, est)
#
#             ess = 1.0 / tf.reduce_sum(tf.square(weights), axis=1)
#             ess_history = ess_history.write(t, ess)
#
#         return tf.transpose(est_states.stack(), [1, 0, 2]), tf.transpose(ess_history.stack(), [1, 0])


if __name__ == "__main__":
    print("Testing classical_filters/particle_filters.py...")


    # --- Dummy Resampler ---
    class DummySystematicResample(tf.Module):
        @tf.function
        def resample(self, p, w):
            n = tf.shape(p)[1]
            new_w = tf.ones_like(w) / tf.cast(n, tf.float32)
            return p, new_w


    # --- Create a Dummy SSM for the PF ---
    class DummyPF_SSM(tf.Module):
        def __init__(self):
            super().__init__()
            self.Q = tf.constant([[0.01]])  # stddev = 0.1
            self.R = tf.constant([[0.1]])

        @tf.function
        def f_fn(self, p, t=None):
            return p * 0.9

        @tf.function
        def h_fn(self, p, t=None):
            return p[..., 0:1]  # Just take the first dimension


    dummy_ssm = DummyPF_SSM()

    pf = ParticleFilter(resampler=DummySystematicResample(), ssm=dummy_ssm, n_particles=50, ess_threshold=0.5)

    # Dummy Data: [Batch=2, Particles=50, Dim=2]
    p_init = tf.random.normal([2, 50, 2])
    w_init = tf.ones([2, 50]) / 50.0
    y_obs = tf.constant([[1.0], [0.5]])  # 2 batches of 1D observations

    p_pred = pf.predict(p_init)
    p_upd, w_upd, est, ess = pf.update(p_pred, w_init, y_obs)
    p_res, w_res = pf.resample_if_needed(p_upd, w_upd, ess)

    print("Estimate Shape:", est.shape)
    print("ESS Shape:", ess.shape)
    print("Resampled Particles Shape:", p_res.shape)
    print("classical_filters/particle_filters.py passed.\n")