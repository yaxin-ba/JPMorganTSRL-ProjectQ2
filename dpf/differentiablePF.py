import tensorflow as tf

# Fallback import for standalone testing
try:
    from core.filter import BaseFilter
except ImportError:
    class BaseFilter(tf.Module):
        def __init__(self, ssm=None, name=None):
            super().__init__(name=name)
            self.ssm = ssm

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


class DifferentiableParticleFilter(tf.Module):
    """
    Generic Differentiable Particle Filter wrapper.
    Connects any valid SSM with a DPF Resampler.
    """

    def __init__(self, ssm, resampler, n_particles=100, name="DPF"):
        super().__init__(name=name)
        self.ssm = ssm
        self.resampler = resampler
        self.N = tf.constant(n_particles, dtype=tf.int32)

    # reduce_retracing=True stops the warning when reusing the class in a loop
    @tf.function(reduce_retracing=True)
    def __call__(self, observations):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]
        D = tf.shape(observations)[2]

        particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
        weights = tf.ones([B, self.N]) / tf.cast(self.N, tf.float32)

        est_states = tf.TensorArray(tf.float32, size=T)
        ess_history = tf.TensorArray(tf.float32, size=T)

        for t in tf.range(T):
            obs = observations[:, t]
            t_f = tf.cast(t + 1, tf.float32)

            if t > 0 and self.resampler is not None:
                particles, weights = self.resampler(particles, weights)

            # Predict
            noise = tf.random.normal(tf.shape(particles)) * tf.sqrt(self.ssm.Q[0, 0])
            particles = self.ssm.f_fn(particles, t_f) + noise

            # Update Likelihood
            pred_obs = self.ssm.h_fn(particles, t_f)
            dist = tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs, 1)), axis=2)
            log_lik = -0.5 * dist / self.ssm.R[0, 0]

            # --- Log-Domain Normalization ---
            log_w = tf.math.log(weights + 1e-16) + log_lik
            lse = tf.reduce_logsumexp(log_w, axis=1, keepdims=True)
            weights = tf.exp(log_w - lse)

            # Safety clamp to prevent pure zeroes
            weights = tf.where(weights < 1e-10, tf.ones_like(weights) * 1e-10, weights)
            weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

            est = tf.reduce_sum(tf.expand_dims(weights, 2) * particles, axis=1)
            est_states = est_states.write(t, est)

            ess = 1.0 / tf.reduce_sum(tf.square(weights), axis=1)
            ess_history = ess_history.write(t, ess)

        return tf.transpose(est_states.stack(), [1, 0, 2]), tf.transpose(ess_history.stack(), [1, 0])


class Differentiable_Li17EDH_Filter(tf.Module):
    """Li17 EDH (PF-PF) Continuous-Time Flow Filter for System Identification."""

    def __init__(self, ssm, resampler, dt=1.0 / 15.0):
        super().__init__()
        self.ssm = ssm
        self.resampler = resampler
        self.dt = dt

    @tf.function
    def get_jacobian_h(self, x, t_f):
        """Automatically calculates the observation Jacobian for ANY state-space model."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            h_x = self.ssm.h_fn(x, t_f)
        return tape.gradient(h_x, x)

    def flow_step(self, x_flow, x_aux, P_pred, z, lam, R, t_f):
        H = self.get_jacobian_h(x_aux, t_f)
        S_inv = 1.0 / (R + lam * (H ** 2) * P_pred + 1e-8)
        A = -0.5 * P_pred * H * S_inv * H
        b = (1.0 + 2.0 * lam * A) * ((1.0 + lam * A) * P_pred * H / R * (
                tf.expand_dims(z, 1) - (self.ssm.h_fn(x_aux, t_f) - H * x_aux)) + A * x_aux)
        return x_flow + self.dt * (A * x_flow + b), x_aux + self.dt * (A * x_aux + b)

    @tf.function(reduce_retracing=True)
    def __call__(self, observations, log_Q, log_R, N=50):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]

        Q = tf.exp(log_Q)
        R = tf.exp(log_R)

        x = tf.random.normal([B, N, 1]) * tf.sqrt(Q)
        weights = tf.ones([B, N]) / tf.cast(N, tf.float32)
        log_lik_total = tf.constant(0.0)

        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(weights, tf.TensorShape([None, None])),
                                  (x, tf.TensorShape([None, None, None])),
                                  (log_lik_total, tf.TensorShape([]))]
            )
            t_f = tf.cast(t + 1, tf.float32)

            if t > 0:
                x, weights = self.resampler(x, weights)

            # Use generic SSM transition
            mu_det = self.ssm.f_fn(x, t_f)
            x_pred = mu_det + tf.random.normal(tf.shape(x)) * tf.sqrt(Q)
            P_pred = tf.math.reduce_variance(x_pred, axis=1, keepdims=True) + 1e-4

            # Flow
            x_f, x_a = x_pred, mu_det
            for k in range(15):
                x_f, x_a = self.flow_step(x_f, x_a, P_pred, observations[:, t], float(k) * self.dt, R, t_f)

            # Likelihood using generic SSM observation
            obs_t = tf.expand_dims(observations[:, t], 1)
            log_l = -0.5 * tf.square(obs_t - self.ssm.h_fn(x_f, t_f)) / R
            log_pr = -0.5 * (tf.square(x_f - mu_det) - tf.square(x_pred - mu_det)) / Q
            log_w = tf.math.log(weights + 1e-16) + tf.squeeze(log_l + log_pr, -1)

            w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
            weights = tf.exp(log_w - w_max)
            w_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
            weights = weights / (w_sum + 1e-16)
            log_lik_total += tf.reduce_mean(tf.squeeze(w_max) + tf.math.log(tf.squeeze(w_sum) + 1e-16))

            est = tf.reduce_sum(weights[:, :, tf.newaxis] * x_f, axis=1)
            est_states = est_states.write(t, est)
            x = x_f

        return tf.transpose(est_states.stack(), [1, 0, 2]), log_lik_total



class Differentiable_Li17_Filter(tf.Module):
    """
    Invertible PF-PF of Li (2017) combined with Corenflos (2021) OT Resampling.
    Tracks the Jacobian determinant of the continuous flow for exact weight calculation.
    """

    def __init__(self, ssm, resampler, dt=1.0 / 15.0):
        super().__init__()
        self.ssm = ssm
        self.resampler = resampler
        self.dt = dt

    @tf.function
    def get_jacobian_h(self, x, t_f):
        with tf.GradientTape() as tape:
            tape.watch(x)
            h_x = self.ssm.h_fn(x, t_f)
        return tape.gradient(h_x, x)

    def flow_step(self, x_flow, x_aux, P_pred, z, lam, R, t_f):
        H = self.get_jacobian_h(x_aux, t_f)
        S_inv = 1.0 / (R + lam * (H ** 2) * P_pred + 1e-8)

        # 'A' represents the derivative of the flow map
        A = -0.5 * P_pred * H * S_inv * H

        obs_err = tf.expand_dims(z, 1) - (self.ssm.h_fn(x_aux, t_f) - H * x_aux)
        b = (1.0 + 2.0 * lam * A) * ((1.0 + lam * A) * P_pred * H / R * obs_err + A * x_aux)

        x_flow_new = x_flow + self.dt * (A * x_flow + b)
        x_aux_new = x_aux + self.dt * (A * x_aux + b)

        # --- Li (2017) Core Contribution: The Flow Jacobian ---
        # The log-determinant of the Jacobian simply evolves as the integral of A(lam)
        d_log_J = A * self.dt

        return x_flow_new, x_aux_new, d_log_J

    @tf.function(reduce_retracing=True)
    def __call__(self, observations, log_Q, log_R, N=50):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]

        Q = tf.exp(log_Q)
        R = tf.exp(log_R)

        x = tf.random.normal([B, N, 1]) * tf.sqrt(Q)
        weights = tf.ones([B, N]) / tf.cast(N, tf.float32)
        log_lik_total = tf.constant(0.0)

        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(weights, tf.TensorShape([None, None])),
                                  (x, tf.TensorShape([None, None, None])),
                                  (log_lik_total, tf.TensorShape([]))]
            )
            t_f = tf.cast(t + 1, tf.float32)

            # Corenflos (2021) Entropy-Regularized OT Resampling
            if t > 0:
                x, weights = self.resampler(x, weights)

            mu_det = self.ssm.f_fn(x, t_f)
            x_pred = mu_det + tf.random.normal(tf.shape(x)) * tf.sqrt(Q)
            P_pred = tf.math.reduce_variance(x_pred, axis=1, keepdims=True) + 1e-4

            x_f, x_a = x_pred, mu_det
            log_J = tf.zeros([B, N, 1])  # Initialize Jacobian log-determinant

            for k in range(15):
                lam = float(k) * self.dt
                x_f, x_a, d_log_J = self.flow_step(x_f, x_a, P_pred, observations[:, t], lam, R, t_f)

                # Accumulate the volume change caused by the flow
                log_J += d_log_J

            # --- Exact Invertible Weight Update ---
            obs_t = tf.expand_dims(observations[:, t], 1)
            log_l = -0.5 * tf.square(obs_t - self.ssm.h_fn(x_f, t_f)) / R
            log_pr = -0.5 * (tf.square(x_f - mu_det) - tf.square(x_pred - mu_det)) / Q

            # log(w) = log(w_old) + log(Likelihood) + log(Prior Ratio) + log(Jacobian Determinant)
            log_w = tf.math.log(weights + 1e-16) + tf.squeeze(log_l + log_pr + log_J, -1)

            w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
            weights = tf.exp(log_w - w_max)
            w_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
            weights = weights / (w_sum + 1e-16)

            log_lik_total += tf.reduce_mean(tf.squeeze(w_max) + tf.math.log(tf.squeeze(w_sum) + 1e-16))

            est = tf.reduce_sum(weights[:, :, tf.newaxis] * x_f, axis=1)
            est_states = est_states.write(t, est)
            x = x_f

        return tf.transpose(est_states.stack(), [1, 0, 2]), log_lik_total
