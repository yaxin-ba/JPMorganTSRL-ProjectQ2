import tensorflow as tf
import math


class LinearGaussianSSM(tf.Module):
    """
    Example 1: Linear Gaussian model.
    X_n = A X_{n-1} + B V_n.
    Y_n = C X_n + D W_n.
    """

    def __init__(self, A, B, C, D):
        super().__init__(name="LGSSM")
        self.A = tf.constant(A, dtype=tf.float32)
        self.B = tf.constant(B, dtype=tf.float32)
        self.C = tf.constant(C, dtype=tf.float32)
        self.D = tf.constant(D, dtype=tf.float32)
        self.Q = tf.linalg.matmul(self.B, self.B, transpose_b=True)
        self.R = tf.linalg.matmul(self.D, self.D, transpose_b=True)

    def generate_data(self, N, x0):
        X_true, Y_obs = [x0], []
        x_curr = x0

        for _ in range(N):
            v = tf.random.normal([tf.shape(self.B)[1]])
            w = tf.random.normal([tf.shape(self.D)[1]])

            x_next = tf.linalg.matvec(self.A, x_curr) + tf.linalg.matvec(self.B, v)
            y_curr = tf.linalg.matvec(self.C, x_next) + tf.linalg.matvec(self.D, w)

            X_true.append(x_next)
            Y_obs.append(y_curr)
            x_curr = x_next

        return tf.stack(X_true[1:]), tf.stack(Y_obs)


class StochasticVolatilitySSM(tf.Module):
    """
    Example 2: Stochastic Volatility (SV) model.
    X_n = \alpha X_{n-1} + \sigma V_n.
    Y_n = \beta \exp(X_n/2) W_n.
    (We use the Y^2 transform for EKF/UKF compatibility as in your previous NumPy code)
    """

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5):
        super().__init__(name="SV_Model")
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.Q = tf.reshape(self.sigma ** 2, [1, 1])
        self.R = tf.constant([[1e-6]], dtype=tf.float32)  # Base dynamic R

    def generate_data(self, N, x0):
        X_true, Z_obs = [x0], []
        x_curr = x0

        for _ in range(N):
            v = tf.random.normal([])
            w = tf.random.normal([])

            x_next = self.alpha * x_curr + self.sigma * v
            y_curr = self.beta * tf.exp(x_next / 2.0) * w

            X_true.append(tf.reshape(x_next, [1]))
            Z_obs.append(tf.reshape(y_curr ** 2, [1]))  # Z_n = Y_n^2
            x_curr = x_next

        return tf.stack(X_true[1:]), tf.stack(Z_obs)

    @tf.function
    def f_fn(self, x, t=None): return self.alpha * x

    @tf.function
    def jacob_f_fn(self, x, t=None): return tf.reshape(self.alpha, [1, 1])

    @tf.function
    def h_fn(self, x, t=None): return (self.beta ** 2) * tf.exp(x)

    @tf.function
    def jacob_h_fn(self, x, t=None): return tf.reshape((self.beta ** 2) * tf.exp(x), [1, 1])

class LGSSM_SSM(tf.Module):
    def __init__(self):
        super().__init__(name="LGSSM")
        self.theta = tf.Variable(0.5, dtype=tf.float32)
        self.Q = tf.constant([[0.01]], dtype=tf.float32) # std=0.1
        self.R = tf.constant([[0.01]], dtype=tf.float32) # std=0.1

    @tf.function
    def f_fn(self, x, t=None):
        return self.theta * x

    @tf.function
    def h_fn(self, x, t=None):
        return x

class UNGM_SSM(tf.Module):
    """
    Example 3: Univariate Non-linear Growth Model.
    X_n = X_{n-1}/2 + 25 X_{n-1}/(1+X_{n-1}^2) + 8 \cos(1.2n) + V_n.
    Y_n = X_n^2/20 + W_n.
    """

    def __init__(self, sigma_v_sq=10.0, sigma_w_sq=1.0):
        super().__init__(name="UNGM")
        self.Q = tf.constant([[sigma_v_sq]], dtype=tf.float32)
        self.R = tf.constant([[sigma_w_sq]], dtype=tf.float32)

    def generate_data(self, N, x0):
        X_true, Y_obs = [x0], []
        x_curr = x0

        for n in range(1, N + 1):
            n_f = tf.cast(n, tf.float32)
            v = tf.random.normal([], stddev=tf.sqrt(self.Q[0, 0]))
            w = tf.random.normal([], stddev=tf.sqrt(self.R[0, 0]))

            term1 = x_curr / 2.0
            term2 = 25.0 * x_curr / (1.0 + x_curr ** 2)
            term3 = 8.0 * tf.cos(1.2 * n_f)

            x_next = term1 + term2 + term3 + v
            y_curr = (x_next ** 2) / 20.0 + w

            X_true.append(tf.reshape(x_next, [1]))
            Y_obs.append(tf.reshape(y_curr, [1]))
            x_curr = x_next

        return tf.stack(X_true[1:]), tf.stack(Y_obs)

    @tf.function
    def f_fn(self, x, t):
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t)

    @tf.function
    def jacob_f_fn(self, x, t):
        term1 = 0.5
        term2 = 25.0 * (1.0 - x ** 2) / tf.square(1.0 + x ** 2)
        return tf.reshape(term1 + term2, [1, 1])

    @tf.function
    def h_fn(self, x, t=None): return (x ** 2) / 20.0

    @tf.function
    def jacob_h_fn(self, x, t=None): return tf.reshape(x / 10.0, [1, 1])


# # Li17
# import tensorflow as tf
#
# class AcousticTrackingSSM(tf.Module):
#     def __init__(self):
#         super().__init__(name="AcousticTracking")
#
#         dt = 1.0
#         F_block = tf.constant([
#             [1.0, 0.0, dt, 0.0],
#             [0.0, 1.0, 0.0, dt],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0]
#         ], dtype=tf.float32)
#
#         Q_block = tf.constant([
#             [3.0, 0.0, 0.1, 0.0],
#             [0.0, 3.0, 0.0, 0.1],
#             [0.1, 0.0, 0.03, 0.0],
#             [0.0, 0.1, 0.0, 0.03]
#         ], dtype=tf.float32)
#
#         F_op = tf.linalg.LinearOperatorFullMatrix(F_block)
#         Q_op = tf.linalg.LinearOperatorFullMatrix(Q_block)
#
#         self.F = tf.linalg.LinearOperatorBlockDiag([F_op] * 4).to_dense()
#         self.Q = tf.linalg.LinearOperatorBlockDiag([Q_op] * 4).to_dense()
#
#         grid = tf.linspace(0.0, 40.0, 5)
#         X, Y = tf.meshgrid(grid, grid)
#         self.sensors = tf.reshape(tf.stack([X, Y], axis=-1), [-1, 2])
#         self.R = tf.eye(25, dtype=tf.float32) * 0.01
#
#     @tf.function
#     def f_fn(self, x, t=None):
#         return tf.linalg.matvec(self.F, x)
#
#     @tf.function
#     def jacob_f_fn(self, x, t=None):
#         batch_shape = tf.shape(x)[:-1]
#         return tf.broadcast_to(self.F, tf.concat([batch_shape, [16, 16]], axis=0))
#
#     @tf.function
#     def h_fn(self, x, t=None):
#         z = tf.zeros(tf.concat([tf.shape(x)[:-1], [25]], axis=0), dtype=tf.float32)
#         for c in range(4):
#             pos_c = x[..., c * 4: c * 4 + 2]
#             # Native broadcast avoids adding batch dims to single observations
#             pos_exp = tf.expand_dims(pos_c, axis=-2)
#             diff_c = pos_exp - self.sensors
#             dist_c = tf.norm(diff_c, axis=-1)
#             z += 10.0 / (dist_c + 0.1)
#         return z
#
#     @tf.function
#     def jacob_h_fn(self, x, t=None):
#         cols = []
#         for c in range(4):
#             pos_c = x[..., c * 4: c * 4 + 2]
#             pos_exp = tf.expand_dims(pos_c, axis=-2)
#             diff_c = pos_exp - self.sensors
#             dist_c = tf.norm(diff_c, axis=-1)
#
#             factor = -10.0 / (tf.square(dist_c + 0.1) * (dist_c + 1e-9))
#             grad_pos_c = tf.expand_dims(factor, -1) * diff_c
#             grad_vel_c = tf.zeros_like(grad_pos_c)
#
#             cols.extend([grad_pos_c, grad_vel_c])
#
#         return tf.concat(cols, axis=-1)

# # Dai22-Li17
# class SparseAngleTrackingSSM(tf.Module):
#     def __init__(self):
#         super().__init__(name="SparseAngleTracking")
#         self.sensors = tf.constant([[-150.0, 0.0], [150.0, 0.0]], dtype=tf.float32)
#         self.R_val = 0.0005
#         self.R = tf.eye(2, dtype=tf.float32) * self.R_val
#         self.dt = 1.0
#
#         self.F = tf.constant([
#             [1.0, 0.0, self.dt, 0.0],
#             [0.0, 1.0, 0.0, self.dt],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0]
#         ], dtype=tf.float32)
#
#         self.Q = tf.eye(4, dtype=tf.float32) * (0.2 ** 2)
#
#     @tf.function
#     def f_fn(self, x, t=None):
#         return tf.linalg.matvec(self.F, x)
#
#     @tf.function
#     def jacob_f_fn(self, x, t=None):
#         batch_shape = tf.shape(x)[:-1]
#         return tf.broadcast_to(self.F, tf.concat([batch_shape, [4, 4]], axis=0))
#
#     @tf.function
#     def h_fn(self, x, t=None):
#         px = x[..., 0:1]
#         py = x[..., 1:2]
#         sx = self.sensors[:, 0]
#         sy = self.sensors[:, 1]
#
#         dx = px - sx
#         dy = py - sy
#         return tf.math.atan2(dy, dx)
#
#     @tf.function
#     def jacob_h_fn(self, x, t=None):
#         px = x[..., 0:1]
#         py = x[..., 1:2]
#         sx = self.sensors[:, 0]
#         sy = self.sensors[:, 1]
#
#         dx = px - sx
#         dy = py - sy
#         r2 = tf.square(dx) + tf.square(dy) + 1e-9
#
#         H_px = -dy / r2
#         H_py = dx / r2
#         H_vx = tf.zeros_like(H_px)
#         H_vy = tf.zeros_like(H_px)
#
#         return tf.stack([H_px, H_py, H_vx, H_vy], axis=-1)

if __name__ == "__main__":
    print("Testing models/benchmarks.py...")
    lg_model = LinearGaussianSSM([[1.0]], [[1.0]], [[1.0]], [[0.1]])
    x_lg, y_lg = lg_model.generate_data(10, tf.constant([0.0]))

    sv_model = StochasticVolatilitySSM()
    x_sv, y_sv = sv_model.generate_data(10, tf.constant(0.0))

    ungm = UNGM_SSM()
    x_un, y_un = ungm.generate_data(10, tf.constant(0.1))
    print("All models generated synthetic data successfully.\n")