import tensorflow as tf

# Fallback import for standalone testing
try:
    from core.filter import BaseFilter
except ImportError:
    class BaseFilter(tf.Module):
        def __init__(self, ssm=None, name=None):
            super().__init__(name=name)
            self.ssm = ssm


class KalmanFilter(BaseFilter):
    def __init__(self, ssm=None, update_type='joseph', name="KalmanFilter"):
        super().__init__(ssm, name=name)
        self.update_type = update_type

    @tf.function
    def predict(self, x_hat, P, A, Q):
        x_hat_prior = tf.linalg.matvec(A, x_hat)
        P_prior = tf.linalg.matmul(A, tf.linalg.matmul(P, A, transpose_b=True)) + Q
        return x_hat_prior, P_prior

    @tf.function
    def update(self, x_hat_prior, P_prior, y, C, R):
        I = tf.eye(tf.shape(P_prior)[0], dtype=P_prior.dtype)

        S = tf.linalg.matmul(C, tf.linalg.matmul(P_prior, C, transpose_b=True)) + R
        S_inv = tf.linalg.inv(S)

        K = tf.linalg.matmul(P_prior, tf.linalg.matmul(C, S_inv, transpose_a=True))
        innov = y - tf.linalg.matvec(C, x_hat_prior)
        x_hat_post = x_hat_prior + tf.linalg.matvec(K, innov)

        if self.update_type == 'joseph':
            I_KC = I - tf.linalg.matmul(K, C)
            term1 = tf.linalg.matmul(I_KC, tf.linalg.matmul(P_prior, I_KC, transpose_b=True))
            term2 = tf.linalg.matmul(K, tf.linalg.matmul(R, K, transpose_b=True))
            P_post = term1 + term2
        else:
            P_post = tf.linalg.matmul(I - tf.linalg.matmul(K, C), P_prior)

        return x_hat_post, P_post


class ExtendedKalmanFilter(BaseFilter):
    def __init__(self, ssm=None, name="EKF"):
        super().__init__(ssm, name=name)

    @tf.function
    def predict(self, x_hat, P, t=None):
        x_hat_prior = self.ssm.f_fn(x_hat, t)
        F = self.ssm.jacob_f_fn(x_hat, t)
        P_prior = tf.linalg.matmul(F, tf.linalg.matmul(P, F, transpose_b=True)) + self.ssm.Q
        return x_hat_prior, P_prior

    @tf.function
    def update(self, x_hat_prior, P_prior, y, t=None, R_dyn=None):
        I = tf.eye(tf.shape(P_prior)[0], dtype=P_prior.dtype)
        R = R_dyn if R_dyn is not None else self.ssm.R

        y_pred = self.ssm.h_fn(x_hat_prior, t)
        H = self.ssm.jacob_h_fn(x_hat_prior, t)

        S = tf.linalg.matmul(H, tf.linalg.matmul(P_prior, H, transpose_b=True)) + R
        S_inv = tf.linalg.inv(S)

        K = tf.linalg.matmul(P_prior, tf.linalg.matmul(H, S_inv, transpose_a=True))
        innov = y - y_pred

        x_hat_post = x_hat_prior + tf.linalg.matvec(K, innov)
        P_post = tf.linalg.matmul(I - tf.linalg.matmul(K, H), P_prior)

        return x_hat_post, P_post


class UnscentedKalmanFilter(BaseFilter):
    def __init__(self, ssm=None, alpha=1e-3, beta=2.0, kappa=0.0, name="UKF"):
        super().__init__(ssm, name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.kappa = tf.constant(kappa, dtype=tf.float32)

    @tf.function
    def get_sigma_points_and_weights(self, x, P):
        n = tf.cast(tf.shape(x)[0], tf.float32)
        lam = self.alpha ** 2 * (n + self.kappa) - n

        Wc = tf.fill([2 * tf.cast(n, tf.int32) + 1], 0.5 / (n + lam))
        Wm = tf.fill([2 * tf.cast(n, tf.int32) + 1], 0.5 / (n + lam))

        Wc = tf.tensor_scatter_nd_update(Wc, [[0]], [lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)])
        Wm = tf.tensor_scatter_nd_update(Wm, [[0]], [lam / (n + lam)])

        jitter = tf.eye(tf.shape(P)[0], dtype=tf.float32) * 1e-6
        L = tf.linalg.cholesky(P + jitter)
        scaled_L = tf.sqrt(n + lam) * L

        sigmas = tf.concat([
            tf.expand_dims(x, 0),
            x + tf.transpose(scaled_L),
            x - tf.transpose(scaled_L)
        ], axis=0)

        return sigmas, Wm, Wc

    @tf.function
    def predict(self, x_hat, P, t=None):
        sigmas, Wm, Wc = self.get_sigma_points_and_weights(x_hat, P)

        sigmas_pred = self.ssm.f_fn(sigmas, t)

        x_hat_prior = tf.tensordot(Wm, sigmas_pred, axes=1)

        diff = sigmas_pred - x_hat_prior
        P_prior = tf.tensordot(Wc, tf.einsum('ij,ik->ijk', diff, diff), axes=1) + self.ssm.Q

        return x_hat_prior, P_prior, sigmas_pred

    @tf.function
    def update(self, x_hat_prior, P_prior, sigmas_pred, y, t=None, R_dyn=None):
        _, Wm, Wc = self.get_sigma_points_and_weights(x_hat_prior, P_prior)
        R = R_dyn if R_dyn is not None else self.ssm.R

        Z_sigmas = self.ssm.h_fn(sigmas_pred, t)
        z_mean = tf.tensordot(Wm, Z_sigmas, axes=1)

        z_diff = Z_sigmas - z_mean
        x_diff = sigmas_pred - x_hat_prior

        S = tf.tensordot(Wc, tf.einsum('ij,ik->ijk', z_diff, z_diff), axes=1) + R
        S_inv = tf.linalg.inv(S)

        Pxz = tf.tensordot(Wc, tf.einsum('ij,ik->ijk', x_diff, z_diff), axes=1)

        K = tf.linalg.matmul(Pxz, S_inv)
        innov = y - z_mean

        x_hat_post = x_hat_prior + tf.linalg.matvec(K, innov)
        P_post = P_prior - tf.linalg.matmul(K, tf.linalg.matmul(S, K, transpose_b=True))

        return x_hat_post, P_post


if __name__ == "__main__":
    print("Testing classical_filters/kalman_filters.py components...")


    # --- Create a Dummy SSM to match the architecture ---
    class DummySSM(tf.Module):
        def __init__(self):
            super().__init__()
            self.Q = tf.eye(2, dtype=tf.float32) * 0.1
            self.R = tf.constant([[0.5]], dtype=tf.float32)

        @tf.function
        def f_fn(self, st, t=None):
            return st * 1.1

        @tf.function
        def jacob_f_fn(self, st, t=None):
            return tf.eye(2) * 1.1

        @tf.function
        def h_fn(self, st, t=None):
            # Expand dims to support batched inputs from UKF sigma points
            if len(st.shape) == 1:
                return tf.expand_dims(st[0], 0)
            return st[..., 0:1]

        @tf.function
        def jacob_h_fn(self, st, t=None):
            return tf.constant([[1.0, 0.0]])


    dummy_ssm = DummySSM()

    # --- Shared Test Data ---
    x = tf.constant([0.0, 0.0], dtype=tf.float32)
    P = tf.eye(2, dtype=tf.float32)
    y = tf.constant([1.2], dtype=tf.float32)

    # --- 1. Test KF ---
    kf = KalmanFilter(update_type='joseph')
    A = tf.constant([[1.0, 0.1], [0.0, 1.0]])
    C = tf.constant([[1.0, 0.0]])

    x_p, P_p = kf.predict(x, P, A, dummy_ssm.Q)
    x_u, P_u = kf.update(x_p, P_p, y, C, dummy_ssm.R)
    print("KF Update Shape:", x_u.shape, P_u.shape)

    # --- 2. Test EKF ---
    ekf = ExtendedKalmanFilter(ssm=dummy_ssm)
    x_pe, P_pe = ekf.predict(x, P)
    x_ue, P_ue = ekf.update(x_pe, P_pe, y)
    print("EKF Update Shape:", x_ue.shape, P_ue.shape)

    # --- 3. Test UKF ---
    ukf = UnscentedKalmanFilter(ssm=dummy_ssm)
    x_pu, P_pu, sigs = ukf.predict(x, P)
    x_uu, P_uu = ukf.update(x_pu, P_pu, sigs, y)
    print("UKF Update Shape:", x_uu.shape, P_uu.shape)

    print("classical_filters/kalman_filters.py passed.\n")