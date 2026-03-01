import tensorflow as tf
import time
import matplotlib.pyplot as plt

from flows.edh_ledh import ParticleFlowParticleFilter
from benchmarks_li17 import AcousticTrackingSSM
from resamplers.resamplers import SystematicResampler

#
# class SystematicResampler(tf.Module):
#     """Pure-TF Systematic Resampler matching the NumPy reference"""
#
#     @tf.function
#     def resample(self, particles, weights):
#         N = tf.shape(particles)[0]
#         N_f = tf.cast(N, tf.float32)
#
#         positions = (tf.range(N_f) + tf.random.uniform([])) / N_f
#         cumulative_sum = tf.cumsum(weights)
#
#         indices = tf.searchsorted(cumulative_sum, positions, side='right')
#         indices = tf.clip_by_value(indices, 0, N - 1)
#
#         new_particles = tf.gather(particles, indices)
#         new_weights = tf.ones_like(weights) / N_f
#         return new_particles, new_weights


def run_acoustic_tracking(n_particles = 200, n_steps = 40):
    print("--- Running Ex 1: Multi-Target Acoustic Tracking (EDH vs LEDH) ---")
    tf.random.set_seed(36)

    # n_steps = 40
    # n_particles = 200
    ssm = AcousticTrackingSSM()

    x_true_init = tf.constant([
        12.0, 6.0, 0.001, 0.001,
        32.0, 32.0, -0.001, -0.005,
        20.0, 13.0, -0.1, 0.01,
        15.0, 35.0, 0.002, 0.002
    ], dtype=tf.float32)

    X_true, Z_obs = [x_true_init], []
    x_curr = x_true_init

    for _ in range(n_steps):
        L_Q = tf.linalg.cholesky(ssm.Q)
        v_k = tf.linalg.matvec(L_Q, tf.random.normal([16]))
        x_next = ssm.f_fn(x_curr) + v_k
        w_k = tf.random.normal([25], stddev=0.1)
        z_curr = ssm.h_fn(x_next) + w_k

        X_true.append(x_next)
        Z_obs.append(z_curr)
        x_curr = x_next

    X_true = tf.stack(X_true[1:])
    Z_obs = tf.stack(Z_obs)

    resampler = SystematicResampler()
    edh_filter = ParticleFlowParticleFilter(resampler=resampler, ssm=ssm, mode='EDH', n_particles=n_particles)
    ledh_filter = ParticleFlowParticleFilter(resampler=resampler, ssm=ssm, mode='LEDH', n_particles=n_particles)

    # Initialize with uniform variance (10.0) mirroring the reference NumPy code
    init_cov = tf.eye(16, dtype=tf.float32) * 10.0
    L_init = tf.linalg.cholesky(init_cov)
    p0 = x_true_init + tf.linalg.matmul(tf.random.normal([n_particles, 16]), L_init, transpose_b=True)

    p_curr_edh, p_curr_ledh = tf.identity(p0), tf.identity(p0)
    w_curr_edh, w_curr_ledh = tf.ones([n_particles]) / float(n_particles), tf.ones([n_particles]) / float(n_particles)
    P_curr_edh, P_curr_ledh = tf.identity(init_cov), tf.identity(init_cov)

    est_history_edh, est_history_ledh = [], []

    print("Tracking in progress...")
    start_time = time.perf_counter()

    for n in range(n_steps):
        if n % 10 == 0: print(f"  Step {n}/{n_steps}")
        z_k = Z_obs[n]

        # --- EDH ---
        eta_0_edh, eta_bar_0_edh, P_pred_edh = edh_filter.predict(p_curr_edh, P_curr_edh)
        p_curr_edh, w_curr_edh, x_est_edh, P_curr_edh, ess_edh = edh_filter.update(
            eta_0_edh, eta_bar_0_edh, P_pred_edh, p_curr_edh, w_curr_edh, z_k
        )
        p_curr_edh, w_curr_edh = edh_filter.resample_if_needed(p_curr_edh, w_curr_edh, ess_edh)
        est_history_edh.append(x_est_edh)

        # --- LEDH ---
        eta_0_ledh, eta_bar_0_ledh, P_pred_ledh = ledh_filter.predict(p_curr_ledh, P_curr_ledh)
        p_curr_ledh, w_curr_ledh, x_est_ledh, P_curr_ledh, ess_ledh = ledh_filter.update(
            eta_0_ledh, eta_bar_0_ledh, P_pred_ledh, p_curr_ledh, w_curr_ledh, z_k
        )
        p_curr_ledh, w_curr_ledh = ledh_filter.resample_if_needed(p_curr_ledh, w_curr_ledh, ess_ledh)
        est_history_ledh.append(x_est_ledh)

    print(f"Tracking complete in {(time.perf_counter() - start_time):.2f} seconds.")

    # Visualization
    ht = X_true.numpy()
    he_edh = tf.stack(est_history_edh).numpy()
    he_ledh = tf.stack(est_history_ledh).numpy()
    sensors = ssm.sensors.numpy()

    plt.figure(figsize=(10, 10))
    plt.scatter(sensors[:, 0], sensors[:, 1], marker='s', c='gray', alpha=0.5, label='Sensors')
    colors = ['r', 'g', 'b', 'c']

    plt.plot([], [], 'k-', label='True Trajectory')
    plt.plot([], [], 'k--', linewidth=2, label='LEDH Estimate')
    plt.plot([], [], 'k:', linewidth=2, label='EDH Estimate')

    for c in range(4):
        idx_x, idx_y = c * 4, c * 4 + 1
        plt.plot(ht[:, idx_x], ht[:, idx_y], c=colors[c], ls='-', alpha=0.6)
        plt.plot(he_ledh[:, idx_x], he_ledh[:, idx_y], c=colors[c], ls='--', lw=2)
        plt.plot(he_edh[:, idx_x], he_edh[:, idx_y], c=colors[c], ls=':', lw=2)

    plt.title("Ex 1: Multi-Target Acoustic Tracking (PF-PF LEDH vs EDH)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_acoustic_tracking(n_particles = 200, n_steps = 40)