import tensorflow as tf
import time
import matplotlib.pyplot as plt

from benchmarks_dai22 import SparseAngleTrackingSSM
from flows.edh_ledh import ParticleFlowParticleFilter
from utils.scheduler import ShootingScheduler
from resamplers.resamplers import SystematicResampler

# class SystematicResampler(tf.Module):
#     @tf.function
#     def resample(self, particles, weights):
#         N = tf.shape(particles)[0]
#         N_f = tf.cast(N, tf.float32)
#         positions = (tf.range(N_f) + tf.random.uniform([])) / N_f
#         cumulative_sum = tf.cumsum(weights)
#         indices = tf.searchsorted(cumulative_sum, positions, side='right')
#         indices = tf.clip_by_value(indices, 0, N - 1)
#         return tf.gather(particles, indices), tf.ones_like(weights) / N_f


def get_true_state(t):
    t_f = tf.cast(t, tf.float32)
    freq = tf.constant(0.05, dtype=tf.float32)
    px = 100.0 * tf.math.sin(freq * t_f)
    py = 100.0 * tf.math.sin(2.0 * freq * t_f)
    vx = 100.0 * freq * tf.math.cos(freq * t_f)
    vy = 100.0 * 2.0 * freq * tf.math.cos(2.0 * freq * t_f)
    return tf.stack([px, py, vx, vy])


def run_tracking_scenario(n_particles = 500, n_steps = 50):
    print("--- Running Unified PFPF: Baseline vs Shooting Method ---")
    ssm = SparseAngleTrackingSSM()
    tf.random.set_seed(42)

    P_init_val = tf.linalg.diag([400.0, 400.0, 25.0, 25.0])
    start_state = get_true_state(0)

    # Pre-compute Optimal Schedule utilizing Dai22 Shooting Method
    print("Compiling BVP Shooting Method in TF...")
    M0 = tf.linalg.inv(P_init_val)
    H_jac = tf.squeeze(ssm.jacob_h_fn(tf.expand_dims(start_state, 0)), 0)
    R_inv = tf.linalg.inv(ssm.R)
    # H_h approx equals H^T R^-1 H
    Hh = tf.linalg.matmul(H_jac, tf.linalg.matmul(R_inv, H_jac), transpose_a=True)

    scheduler = ShootingScheduler(M0, Hh, mu=0.2)
    optimal_beta_steps = scheduler.get_beta_steps()
    print("Optimal Schedule Extracted.")

    resampler = SystematicResampler()

    # BASELINE: Omit schedule step sizes
    filter_base = ParticleFlowParticleFilter(resampler, ssm, mode='LEDH', n_particles=n_particles)

    # OPTIMAL: Pass derived schedule step sizes
    filter_opt = ParticleFlowParticleFilter(resampler, ssm, mode='LEDH', n_particles=n_particles,
                                            beta_steps=optimal_beta_steps)

    def track(pfpf):
        L_init = tf.linalg.cholesky(P_init_val)
        noise = tf.linalg.matmul(tf.random.normal([n_particles, 4]), L_init, transpose_b=True)
        p_curr = tf.expand_dims(start_state, 0) + noise

        w_curr = tf.ones([n_particles]) / tf.cast(n_particles, tf.float32)
        P_curr = tf.identity(P_init_val)

        traj, errs = [], []

        for t in range(n_steps):
            true_x = get_true_state(t)

            clean_z = ssm.h_fn(tf.expand_dims(true_x, 0))[0]
            z_obs = clean_z + tf.random.normal([2], stddev=tf.sqrt(ssm.R_val))

            eta_0, eta_bar_0, P_pred = pfpf.predict(p_curr, P_curr)
            p_curr, w_curr, x_est, P_curr, ess = pfpf.update(eta_0, eta_bar_0, P_pred, p_curr, w_curr, z_obs)
            p_curr, w_curr = pfpf.resample_if_needed(p_curr, w_curr, ess)

            traj.append(x_est)
            errs.append(tf.linalg.norm(x_est[:2] - true_x[:2]))

        return tf.stack(traj), tf.stack(errs)

    start = time.perf_counter()
    traj_base, err_base = track(filter_base)
    print(f"Baseline (Standard LEDH) Tracked in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    traj_opt, err_opt = track(filter_opt)
    print(f"Optimal (Shooting) Tracked in {time.perf_counter() - start:.2f}s")

    true_path_plot = tf.stack([get_true_state(t) for t in range(n_steps)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(true_path_plot[:, 0], true_path_plot[:, 1], 'k-', lw=3, alpha=0.3, label='Truth')
    ax1.plot(traj_base[:, 0], traj_base[:, 1], 'b--', label='Baseline LEDH')
    ax1.plot(traj_opt[:, 0], traj_opt[:, 1], 'r-', lw=2, label='Optimal LEDH (Stiffness Mitigated)')
    ax1.set_title(f"Tracking with Sparse Angle Updates (dt={ssm.dt})")
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    ax2.plot(err_base, 'b--', alpha=0.5, label='Baseline LEDH')
    ax2.plot(err_opt, 'r-', lw=2, label='Optimal LEDH')
    ax2.set_title("RMSE Error (m)")
    ax2.legend()
    ax2.grid(True)

    plt.show()

    print(f"Avg RMSE Baseline: {tf.reduce_mean(err_base):.2f}")
    print(f"Avg RMSE Optimal:  {tf.reduce_mean(err_opt):.2f}")


if __name__ == "__main__":
    run_tracking_scenario(n_particles = 500, n_steps = 50)