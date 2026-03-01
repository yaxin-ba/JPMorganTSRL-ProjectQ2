import tensorflow as tf

try:
    from core.filter import BaseFilter
except ImportError:
    class BaseFilter(tf.Module):
        def __init__(self, ssm=None, name=None):
            super().__init__(name=name)
            self.ssm = ssm


class ParticleFlowParticleFilter(BaseFilter):
    def __init__(self, resampler, ssm=None, mode='LEDH', n_particles=100, ess_threshold=0.5, n_lambda=29, q_step = 1.2,
                 beta_steps=None, name="PFPF"):
        super().__init__(ssm, name=name)
        self.mode = mode
        self.resampler = resampler
        self.N = tf.constant(n_particles, dtype=tf.int32)
        self.N_f = tf.cast(n_particles, tf.float32)
        self.ess_threshold = tf.constant(ess_threshold, dtype=tf.float32)

        # Default Li17 Log-spaced steps
        # n_lambda = 29
        # q_step = 1.2
        eps1 = (1.0 - q_step) / (1.0 - q_step ** n_lambda)
        steps = [eps1 * (q_step ** i) for i in range(n_lambda)]

        # --- STIFFNESS MITIGATION INJECTION ---
        # If optimal beta steps are provided from the Dai22 BVP solver, use them.
        # Otherwise, fall back to the exact standard flow.
        if beta_steps is None:
            self.eps_steps = tf.constant(steps, dtype=tf.float32)
        else:
            self.eps_steps = tf.constant(beta_steps, dtype=tf.float32)

    @tf.function
    def predict(self, particles_prev, P_post, t=None):
        x_mean = tf.reduce_mean(particles_prev, axis=0)
        F = self.ssm.jacob_f_fn(x_mean, t)
        P_pred = tf.linalg.matmul(F, tf.linalg.matmul(P_post, F, transpose_b=True)) + self.ssm.Q

        eta_bar_0 = self.ssm.f_fn(particles_prev, t)
        L = tf.linalg.cholesky(self.ssm.Q)
        noise = tf.linalg.matmul(tf.random.normal(tf.shape(eta_bar_0)), L, transpose_b=True)
        eta_0 = eta_bar_0 + noise

        return eta_0, eta_bar_0, P_pred

    @tf.function
    def _calc_flow_params_single(self, eta_bar, eta_bar_0_anchor, P, R_cov, R_inv, z, lam, t):
        eta_batch = tf.expand_dims(eta_bar, 0)
        H = tf.squeeze(self.ssm.jacob_h_fn(eta_batch, t), 0)
        h_val = tf.squeeze(self.ssm.h_fn(eta_batch, t), 0)

        S = lam * tf.linalg.matmul(H, tf.linalg.matmul(P, H, transpose_b=True)) + R_cov
        S_inv = tf.linalg.pinv(S)

        K_part = tf.linalg.matmul(P, tf.linalg.matmul(H, S_inv, transpose_a=True))
        A = -0.5 * tf.linalg.matmul(K_part, H)

        e = h_val - tf.linalg.matvec(H, eta_bar)

        I_d = tf.eye(tf.shape(P)[0], dtype=tf.float32)
        term1_mat = tf.linalg.matmul(I_d + lam * A, tf.linalg.matmul(P, tf.linalg.matmul(H, R_inv, transpose_a=True)))
        term1 = tf.linalg.matvec(term1_mat, z - e)

        term2 = tf.linalg.matvec(A, eta_bar_0_anchor)
        b = tf.linalg.matvec(I_d + 2.0 * lam * A, term1 + term2)

        return A, b

    @tf.function
    def update(self, eta_0, eta_bar_0_indiv, P_pred, x_prev, weights, z, t=None):
        N = tf.shape(eta_0)[0]
        dim = tf.shape(eta_0)[1]

        R_cov = self.ssm.R
        R_inv = tf.linalg.pinv(R_cov)

        lam = tf.constant(0.0, dtype=tf.float32)
        eta_1 = tf.identity(eta_0)
        log_det_J = tf.zeros([N], dtype=tf.float32)

        if self.mode == 'EDH':
            x_prev_mean = tf.reduce_mean(x_prev, axis=0)
            eta_bar_0_global = self.ssm.f_fn(x_prev_mean, t)
            eta_bar = tf.identity(eta_bar_0_global)
            eta_bar_0_anchor = tf.identity(eta_bar_0_global)
        else:
            eta_bar = tf.identity(eta_bar_0_indiv)
            eta_bar_0_anchor = tf.identity(eta_bar_0_indiv)

        def flow_step(j, lam_curr, e1, e_bar, ldet):
            eps = self.eps_steps[j]
            lam_next = lam_curr + eps

            if self.mode == 'EDH':
                A, b = self._calc_flow_params_single(e_bar, eta_bar_0_anchor, P_pred, R_cov, R_inv, z, lam_next, t)

                drift_1 = tf.linalg.matmul(e1, A, transpose_b=True) + b
                e1_next = e1 + eps * drift_1

                drift_bar = tf.linalg.matvec(A, e_bar) + b
                e_bar_next = e_bar + eps * drift_bar

                ldet_next = ldet
            else:
                A, b = tf.vectorized_map(
                    lambda elems: self._calc_flow_params_single(elems[0], elems[1], P_pred, R_cov, R_inv, z, lam_next,
                                                                t),
                    (e_bar, eta_bar_0_anchor)
                )

                drift_1 = tf.squeeze(tf.linalg.matmul(A, tf.expand_dims(e1, -1)), -1) + b
                e1_next = e1 + eps * drift_1

                drift_bar = tf.squeeze(tf.linalg.matmul(A, tf.expand_dims(e_bar, -1)), -1) + b
                e_bar_next = e_bar + eps * drift_bar

                I_batch = tf.eye(dim, batch_shape=[N])
                _, step_logdet = tf.linalg.slogdet(I_batch + eps * A)
                ldet_next = ldet + step_logdet

            return j + 1, lam_next, e1_next, e_bar_next, ldet_next

        _, _, eta_1, _, log_det_J = tf.while_loop(
            cond=lambda j, *_: j < 29,
            body=flow_step,
            loop_vars=(tf.constant(0), lam, eta_1, eta_bar, log_det_J)
        )

        cov_inv = tf.linalg.pinv(self.ssm.Q)

        diff_1 = eta_1 - eta_bar_0_indiv
        log_p_eta1 = -0.5 * tf.reduce_sum(tf.linalg.matmul(diff_1, cov_inv) * diff_1, axis=-1)

        diff_0 = eta_0 - eta_bar_0_indiv
        log_p_eta0 = -0.5 * tf.reduce_sum(tf.linalg.matmul(diff_0, cov_inv) * diff_0, axis=-1)

        pred_y = self.ssm.h_fn(eta_1, t)
        diff_z = tf.expand_dims(z, 0) - pred_y
        log_liks = -0.5 * tf.reduce_sum(tf.linalg.matmul(diff_z, R_inv) * diff_z, axis=-1)

        if self.mode == 'EDH':
            log_weights = tf.math.log(weights + 1e-300) + log_liks + log_p_eta1 - log_p_eta0
        else:
            log_weights = tf.math.log(weights + 1e-300) + log_liks + log_p_eta1 - log_p_eta0 + log_det_J

        log_weights_max = tf.reduce_max(log_weights, axis=-1, keepdims=True)
        w_unnorm = tf.exp(log_weights - log_weights_max)
        weights_update = w_unnorm / tf.reduce_sum(w_unnorm, axis=-1, keepdims=True)

        x_est = tf.reduce_sum(tf.expand_dims(weights_update, -1) * eta_1, axis=-2)
        ess = 1.0 / tf.reduce_sum(tf.square(weights_update), axis=-1)

        H_final = tf.squeeze(self.ssm.jacob_h_fn(tf.expand_dims(x_est, 0), t), 0)
        S_final = tf.linalg.matmul(H_final, tf.linalg.matmul(P_pred, H_final, transpose_b=True)) + R_cov
        K_final = tf.linalg.matmul(P_pred, tf.linalg.matmul(H_final, tf.linalg.pinv(S_final), transpose_a=True))
        I_mat = tf.eye(dim, dtype=tf.float32)
        P_post = tf.linalg.matmul(I_mat - tf.linalg.matmul(K_final, H_final), P_pred)

        return eta_1, weights_update, x_est, P_post, ess

    @tf.function
    def resample_if_needed(self, particles, weights, ess):
        def do_resample(): return self.resampler.resample(particles, weights)

        def skip_resample(): return particles, weights

        return tf.cond(
            tf.reduce_mean(ess) < (self.ess_threshold * self.N_f),
            true_fn=do_resample, false_fn=skip_resample
        )