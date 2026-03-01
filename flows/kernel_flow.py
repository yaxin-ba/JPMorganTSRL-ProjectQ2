import tensorflow as tf

try:
    from core.filter import BaseFilter
except ImportError:
    class BaseFilter(tf.Module):
        def __init__(self, ssm=None, name=None): super().__init__(name=name)


class MatrixKernelFlow(BaseFilter):
    def __init__(self, ssm=None, n_steps=100, ds=0.01, name="MatrixKernelFlow"):
        super().__init__(ssm, name=name)
        self.n_steps = tf.constant(n_steps, dtype=tf.int32)
        self.ds = tf.constant(ds, dtype=tf.float32)

    @tf.function
    def update(self, particles, y, R_inv, H_idx, B_local):
        """
        Matrix-Valued Kernel Flow Update.

        Args:
            particles: Tensor of shape [N, dim]
            y: Observation Tensor of shape [meas_dim]
            R_inv: Inverse measurement noise covariance [meas_dim, meas_dim]
            H_idx: 1D Tensor of integer indices indicating which dimensions are observed
            B_local: Background covariance preconditioner [dim, dim]
        """
        N = tf.shape(particles)[0]
        dim = tf.shape(particles)[1]
        N_f = tf.cast(N, tf.float32)

        # alpha = 2.0 / N as per the reference implementation
        alpha = 2.0 / N_f

        # Precompute variances for the kernel (diagonal of B_local)
        sigmas_sq = tf.linalg.diag_part(B_local) + 1e-6

        x_s = tf.identity(particles)

        # We compute the source gradients for ALL particles first.
        # This assumes a linear selection matrix H that just picks out H_idx.
        # innov = y - x_s[:, H_idx]
        observed_x = tf.gather(x_s, H_idx, axis=1)
        innovations = tf.expand_dims(y, 0) - observed_x  # [N, meas_dim]

        # gradient of log likelihood with respect to the observed dimensions
        grad_obs = tf.linalg.matmul(innovations, R_inv)  # [N, meas_dim]

        # Scatter the observed gradients back into the full [N, dim] state space
        indices = tf.expand_dims(tf.cast(H_idx, tf.int32), 1)

        # tf.vectorized_map or a loop is needed to scatter for each particle in the batch
        def scatter_grad(g):
            return tf.scatter_nd(indices, g, [dim])

        all_grads = tf.map_fn(scatter_grad, grad_obs)  # [N, dim]

        # Define the pseudo-time step loop
        def step_fn(step, x_curr):
            # x_curr is [N, dim]. We need pairwise differences: x_curr[i] - x_curr[j]
            # diffs shape: [N, N, dim] where diffs[i, j, :] = x_curr[i] - x_curr[j]
            x_i = tf.expand_dims(x_curr, 1)  # [N, 1, dim]
            x_j = tf.expand_dims(x_curr, 0)  # [1, N, dim]
            diffs = x_i - x_j

            # Component-wise Kernel Interactions
            norm_diff_sq = tf.square(diffs) / (alpha * sigmas_sq)
            K_vals = tf.exp(-0.5 * norm_diff_sq)  # [N, N, dim]

            # Divergence of Kernel (Repelling Term)
            div_K = (diffs / (alpha * sigmas_sq)) * K_vals  # [N, N, dim]

            # Broadcast source gradients: [1, N, dim] so it applies to the 'j' dimension
            source_grads_broadcast = tf.expand_dims(all_grads, 0)

            # term1: K_vals * source gradients
            term1 = K_vals * source_grads_broadcast

            # Average over particles (the 1/N sum)
            integral_approx = tf.reduce_mean(term1 + div_K, axis=1)  # [N, dim]

            # Apply Preconditioner B_local
            flow_update = tf.linalg.matmul(integral_approx, B_local)  # [N, dim]

            return step + 1, x_curr + self.ds * flow_update

        # Execute Flow
        _, x_final = tf.while_loop(
            cond=lambda step, _: step < self.n_steps,
            body=step_fn,
            loop_vars=(0, x_s)
        )

        return x_final


if __name__ == "__main__":
    print("Testing flows/kernel_flow.py...")
    # Dummy data to verify graph compilation and tensor shapes
    dim = 4
    N_particles = 20

    kf_flow = MatrixKernelFlow(n_steps=10, ds=0.01)

    particles = tf.random.normal([N_particles, dim])
    y = tf.constant([1.5, -0.5], dtype=tf.float32)
    R_inv = tf.eye(2, dtype=tf.float32)

    # Let's say we observe dimension 0 and dimension 2
    H_idx = tf.constant([0, 2], dtype=tf.int32)
    B_local = tf.eye(dim, dtype=tf.float32) * 2.0

    p_post = kf_flow.update(particles, y, R_inv, H_idx, B_local)

    print("Prior Particles Shape:", particles.shape)
    print("Posterior Particles Shape:", p_post.shape)
    print("flows/kernel_flow.py passed.\n")