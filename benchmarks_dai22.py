import tensorflow as tf


class Dai22StaticExample(tf.Module):
    """
    Replicates the exact static 1-step example from Section 4 of Dai22.
    Target at (4,4), Sensors at (3.5, 0) and (-3.5, 0).
    """

    def __init__(self):
        super().__init__(name="Dai22StaticExample")
        self.P_prior = tf.linalg.diag([1000.0, 2.0])
        self.R_cov = tf.linalg.diag([0.04, 0.04])
        self.x_prior = tf.constant([3.0, 5.0], dtype=tf.float32)
        self.sensors = tf.constant([[3.5, 0.0], [-3.5, 0.0]], dtype=tf.float32)

    @tf.function
    def get_jacobian_and_matrices(self):
        dx1 = self.x_prior[0] - self.sensors[0, 0]
        dy1 = self.x_prior[1] - self.sensors[0, 1]
        r2_1 = tf.square(dx1) + tf.square(dy1)

        dx2 = self.x_prior[0] - self.sensors[1, 0]
        dy2 = self.x_prior[1] - self.sensors[1, 1]
        r2_2 = tf.square(dx2) + tf.square(dy2)

        # H = tf.constant([
        #     [-dy1 / r2_1, dx1 / r2_1],
        #     [-dy2 / r2_2, dx2 / r2_2]
        # ], dtype=tf.float32)
        H = tf.convert_to_tensor([[-dy1 / r2_1, dx1 / r2_1], [-dy2 / r2_2, dx2 / r2_2]], dtype=tf.float32)

        M0 = tf.linalg.inv(self.P_prior)
        R_inv = tf.linalg.inv(self.R_cov)
        Hh = tf.linalg.matmul(H, tf.linalg.matmul(R_inv, H), transpose_a=True)

        return M0, Hh


class SparseAngleTrackingSSM(tf.Module):
    """
    Dynamic 4D Tracking System using Sparse Angle Measurements.
    High process noise with highly informative measurements, prone to stiffness.
    """

    def __init__(self):
        super().__init__(name="SparseAngleTracking")
        self.sensors = tf.constant([[-150.0, 0.0], [150.0, 0.0]], dtype=tf.float32)
        self.R_val = 0.0005
        self.R = tf.eye(2, dtype=tf.float32) * self.R_val
        self.dt = 1.0

        self.F = tf.constant([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=tf.float32)

        self.Q = tf.eye(4, dtype=tf.float32) * (0.2 ** 2)

    @tf.function
    def f_fn(self, x, t=None):
        return tf.linalg.matvec(self.F, x)

    @tf.function
    def jacob_f_fn(self, x, t=None):
        batch_shape = tf.shape(x)[:-1]
        target_shape = tf.concat([batch_shape, tf.constant([4, 4], dtype=tf.int32)], axis=0)
        return tf.broadcast_to(self.F, target_shape)

    @tf.function
    def h_fn(self, x, t=None):
        px = x[..., 0:1]
        py = x[..., 1:2]
        sx = self.sensors[:, 0]
        sy = self.sensors[:, 1]

        dx = px - sx
        dy = py - sy
        return tf.math.atan2(dy, dx)

    @tf.function
    def jacob_h_fn(self, x, t=None):
        px = x[..., 0:1]
        py = x[..., 1:2]
        sx = self.sensors[:, 0]
        sy = self.sensors[:, 1]

        dx = px - sx
        dy = py - sy
        r2 = tf.square(dx) + tf.square(dy) + 1e-9

        H_px = -dy / r2
        H_py = dx / r2
        H_vx = tf.zeros_like(H_px)
        H_vy = tf.zeros_like(H_px)

        return tf.stack([H_px, H_py, H_vx, H_vy], axis=-1)

