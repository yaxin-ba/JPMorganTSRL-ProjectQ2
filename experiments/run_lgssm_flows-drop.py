import tensorflow as tf
import time
import matplotlib.pyplot as plt

from benchmarks import LinearGaussianSSM
from flows.edh_ledh import ParticleFlowParticleFilter
from resamplers.resamplers import SystematicResampler, MultinomialResample


# --- 1. Extend LGSSM to support the refactored architecture ---
class Flow_LGSSM(LinearGaussianSSM):
    def __init__(self, A, B, C, D):
        super().__init__(A, B, C, D)

    @tf.function
    def f_fn(self, x, t=None):
        return tf.linalg.matvec(self.A, x)

    @tf.function
    def jacob_f_fn(self, x, t=None):
        # Jacobian of Ax is just A
        return self.A

    @tf.function
    def h_fn(self, x, t=None):
        return tf.linalg.matvec(self.C, x)

    @tf.function
    def jacob_h_fn(self, x, t=None):
        return self.C


def run_lgssm_flows():
    print("--- Running High-Dim LGSSM Benchmark: EDH vs LEDH ---")
    tf.random.set_seed(42)

    # 2. Setup High-Dimensional Linear System
    dim = 64
    alpha_dyn = 0.9
    N_steps = 15
    N_particles = 200

    # Generate Spatial Covariance Q (Pure TensorFlow)
    # Create 8x8 grid coordinates
    coords = tf.constant([(i, j) for i in range(8) for j in range(8)], dtype=tf.float32)

    # Broadcast to find pairwise squared distances: shape (64, 64)
    diff = tf.expand_dims(coords, 1) - tf.expand_dims(coords, 0)
    dist2 = tf.reduce_sum(tf.square(diff), axis=-1)

    # Compute Q matrix and add jitter to the diagonal
    Q_tf = 3.0 * tf.exp(-dist2 / 20.0) + (tf.eye(dim, dtype=tf.float32) * 0.01)

    # Extract B from Q via Cholesky
    B_tf = tf.linalg.cholesky(Q_tf)

    # Matrices
    A_mat = tf.eye(dim, dtype=tf.float32) * alpha_dyn
    C_mat = tf.eye(dim, dtype=tf.float32)
    D_mat = tf.eye(dim, dtype=tf.float32) * 1.0  # R = I

    lgssm = Flow_LGSSM(A_mat, B_tf, C_mat, D_mat)

    x0 = tf.zeros([dim], dtype=tf.float32)
    X_true, Y_obs = lgssm.generate_data(N_steps, x0)

    # 3. Initialize Filters and Particles
    # edh_filter = ParticleFlowParticleFilter(ssm=lgssm, mode='EDH', n_lambda=29)
    # ledh_filter = ParticleFlowParticleFilter(ssm=lgssm, mode='LEDH', n_lambda=29)
    # 3. Initialize Filters and Particles
    resampler = SystematicResampler()

    edh_filter = ParticleFlowParticleFilter(resampler=resampler, ssm=lgssm, mode='EDH', n_particles=N_particles)
    ledh_filter = ParticleFlowParticleFilter(resampler=resampler, ssm=lgssm, mode='LEDH', n_particles=N_particles)

    # Shared initial prior particles N(0, I)
    p0 = tf.random.normal([N_particles, dim], dtype=tf.float32)
    init_cov = tf.eye(dim, dtype=tf.float32) * 1.0

    p_curr_edh, p_curr_ledh = tf.identity(p0), tf.identity(p0)
    w_curr_edh = tf.ones([N_particles]) / float(N_particles)
    w_curr_ledh = tf.ones([N_particles]) / float(N_particles)

    P_curr_edh, P_curr_ledh = tf.identity(init_cov), tf.identity(init_cov)

    mse_edh, mse_ledh = [], []

    # 4. Tracking Loop
    print("Tracking in progress...")
    start_time = time.perf_counter()

    for n in range(N_steps):
        z_k = Y_obs[n]

        # --- EDH Update ---
        eta_0_edh, eta_bar_0_edh, P_pred_edh = edh_filter.predict(p_curr_edh, P_curr_edh)
        p_curr_edh, w_curr_edh, x_est_edh, P_curr_edh, ess_edh = edh_filter.update(
            eta_0_edh, eta_bar_0_edh, P_pred_edh, p_curr_edh, w_curr_edh, z_k
        )
        p_curr_edh, w_curr_edh = edh_filter.resample_if_needed(p_curr_edh, w_curr_edh, ess_edh)
        mse_edh.append(tf.reduce_mean(tf.square(x_est_edh - X_true[n])))

        # --- LEDH Update ---
        eta_0_ledh, eta_bar_0_ledh, P_pred_ledh = ledh_filter.predict(p_curr_ledh, P_curr_ledh)
        p_curr_ledh, w_curr_ledh, x_est_ledh, P_curr_ledh, ess_ledh = ledh_filter.update(
            eta_0_ledh, eta_bar_0_ledh, P_pred_ledh, p_curr_ledh, w_curr_ledh, z_k
        )
        p_curr_ledh, w_curr_ledh = ledh_filter.resample_if_needed(p_curr_ledh, w_curr_ledh, ess_ledh)
        mse_ledh.append(tf.reduce_mean(tf.square(x_est_ledh - X_true[n])))

    print(f"Tracking complete in {(time.perf_counter() - start_time):.2f} seconds.")

    # 5. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(mse_ledh, '*-', label='PF-PF (LEDH)')
    plt.plot(mse_edh, 'o-', label='PF-PF (EDH)')
    plt.legend()
    plt.title(f"Ex 2: Linear Gaussian MSE ({dim}-Dimensional)")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_lgssm_flows()