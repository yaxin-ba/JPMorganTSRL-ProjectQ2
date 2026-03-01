import tensorflow as tf
import time
import matplotlib.pyplot as plt

from flows.kernel_flow import MatrixKernelFlow


def run_kernel_flow(n_particles=20):
    print("--- Running Matrix Kernel Flow: Marginal Collapse Rescue ---")
    tf.random.set_seed(42)

    # 1. Setup L96-style parameters
    dim = 100

    # Ground truth state
    x_true = tf.random.uniform([dim], dtype=tf.float32) + 8.0

    # Observe every 4th dimension
    H_idx = tf.range(0, dim, 4, dtype=tf.int32)
    num_obs = tf.shape(H_idx)[0]

    # 2. Generate Prior Ensemble and Observation
    prior_particles = x_true + tf.random.normal([n_particles, dim], stddev=1.5)

    # y = H * x_true + noise
    x_true_obs = tf.gather(x_true, H_idx)
    y_obs = x_true_obs + tf.random.normal([num_obs], stddev=0.5)

    # Measurement noise inverse
    R_inv = tf.eye(num_obs, dtype=tf.float32) * (1.0 / (0.5 ** 2))

    # 3. Estimate Background Covariance B_local with localization (Pure TF)
    # Calculate sample covariance: (X - mean)^T @ (X - mean) / (N - 1)
    prior_mean = tf.reduce_mean(prior_particles, axis=0, keepdims=True)
    centered_prior = prior_particles - prior_mean
    raw_cov = tf.matmul(centered_prior, centered_prior, transpose_a=True) / tf.cast(n_particles - 1, tf.float32)

    # Vectorized localization matrix (C_loc)
    coords = tf.range(dim, dtype=tf.float32)
    diff = tf.abs(tf.expand_dims(coords, 1) - tf.expand_dims(coords, 0))
    # Periodic distance: min(|i - j|, dim - |i - j|)
    dist = tf.minimum(diff, tf.cast(dim, tf.float32) - diff)
    C_loc = tf.exp(-tf.square(dist / 4.0))

    # Element-wise multiplication for localization
    B_local = raw_cov * C_loc

    # 4. Run Matrix Kernel Flow
    print("Executing Matrix Kernel Flow (compiling tf.while_loop)...")
    start_time = time.perf_counter()

    # Use 100 pseudo-time steps with ds=0.01 as in the reference
    mkf = MatrixKernelFlow(n_steps=100, ds=0.01)
    post_matrix = mkf.update(prior_particles, y_obs, R_inv, H_idx, B_local)

    print(f"Flow complete in {(time.perf_counter() - start_time):.2f} seconds.")

    # 5. Visualization (Focus on Unobserved x19 vs Observed x20)
    # Since H_idx observes 0, 4, 8, 12, 16, 20...
    # Index 20 is OBSERVED. Index 19 is UNOBSERVED.
    prior_plot = prior_particles.numpy()
    post_plot = post_matrix.numpy()
    x_true_np = x_true.numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(prior_plot[:, 19], prior_plot[:, 20], 'ko', fillstyle='none', markersize=8, label='Prior Ensemble')
    plt.plot(post_plot[:, 19], post_plot[:, 20], 'rx', markersize=8, markeredgewidth=2, label='Posterior Ensemble')

    # Plot truth
    plt.plot(x_true_np[19], x_true_np[20], 'b*', markersize=12, label='True State')

    plt.title("Matrix-Valued Kernel Flow\nDiversity Maintained in Unobserved Dimension")
    plt.xlabel("Unobserved Dimension ($x_{19}$)")
    plt.ylabel("Observed Dimension ($x_{20}$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kernel_flow(n_particles=20)