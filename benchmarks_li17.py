# Li17
import tensorflow as tf

class AcousticTrackingSSM(tf.Module):
    def __init__(self):
        super().__init__(name="AcousticTracking")

        dt = 1.0
        F_block = tf.constant([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=tf.float32)

        Q_block = tf.constant([
            [3.0, 0.0, 0.1, 0.0],
            [0.0, 3.0, 0.0, 0.1],
            [0.1, 0.0, 0.03, 0.0],
            [0.0, 0.1, 0.0, 0.03]
        ], dtype=tf.float32)

        F_op = tf.linalg.LinearOperatorFullMatrix(F_block)
        Q_op = tf.linalg.LinearOperatorFullMatrix(Q_block)

        self.F = tf.linalg.LinearOperatorBlockDiag([F_op] * 4).to_dense()
        self.Q = tf.linalg.LinearOperatorBlockDiag([Q_op] * 4).to_dense()

        grid = tf.linspace(0.0, 40.0, 5)
        X, Y = tf.meshgrid(grid, grid)
        self.sensors = tf.reshape(tf.stack([X, Y], axis=-1), [-1, 2])
        self.R = tf.eye(25, dtype=tf.float32) * 0.01

    @tf.function
    def f_fn(self, x, t=None):
        return tf.linalg.matvec(self.F, x)

    @tf.function
    def jacob_f_fn(self, x, t=None):
        batch_shape = tf.shape(x)[:-1]
        return tf.broadcast_to(self.F, tf.concat([batch_shape, [16, 16]], axis=0))

    @tf.function
    def h_fn(self, x, t=None):
        z = tf.zeros(tf.concat([tf.shape(x)[:-1], [25]], axis=0), dtype=tf.float32)
        for c in range(4):
            pos_c = x[..., c * 4: c * 4 + 2]
            # Native broadcast avoids adding batch dims to single observations
            pos_exp = tf.expand_dims(pos_c, axis=-2)
            diff_c = pos_exp - self.sensors
            dist_c = tf.norm(diff_c, axis=-1)
            z += 10.0 / (dist_c + 0.1)
        return z

    @tf.function
    def jacob_h_fn(self, x, t=None):
        cols = []
        for c in range(4):
            pos_c = x[..., c * 4: c * 4 + 2]
            pos_exp = tf.expand_dims(pos_c, axis=-2)
            diff_c = pos_exp - self.sensors
            dist_c = tf.norm(diff_c, axis=-1)

            factor = -10.0 / (tf.square(dist_c + 0.1) * (dist_c + 1e-9))
            grad_pos_c = tf.expand_dims(factor, -1) * diff_c
            grad_vel_c = tf.zeros_like(grad_pos_c)

            cols.extend([grad_pos_c, grad_vel_c])

        return tf.concat(cols, axis=-1)