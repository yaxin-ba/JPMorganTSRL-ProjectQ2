import tensorflow as tf
import time
import matplotlib.pyplot as plt

from benchmarks import UNGM_SSM
from classical_filters.kalman_filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from classical_filters.particle_filters import ParticleFilter
from resamplers.dresamplers import SoftResample


def run_ungm_comparison(n_particles=200, n_steps=100, q_var=10.0, r_var=1.0):
    print("--- Running UNGM Benchmark: KF vs EKF vs UKF vs PF ---")

    # 1. Setup Data
    tf.random.set_seed(42)

    # Using variances from the reference: sigma_V^2 = 10, sigma_W^2 = 1
    ungm = UNGM_SSM(sigma_v_sq=q_var, sigma_w_sq=r_var)
    x0 = tf.constant([0.1], dtype=tf.float32)

    X_true, Y_obs = ungm.generate_data(n_steps, x0)

    # 2. Filter Initialization
    P0 = tf.constant([[10.0]], dtype=tf.float32)

    # Naive Linear KF (will fail spectacularly on this non-linear system)
    kf = KalmanFilter(ssm=ungm, update_type='standard')
    A_naive = tf.constant([[1.0]], dtype=tf.float32)
    C_naive = tf.constant([[1.0]], dtype=tf.float32)

    ekf = ExtendedKalmanFilter(ssm=ungm)
    ukf = UnscentedKalmanFilter(ssm=ungm, alpha=1.0, beta=0.0, kappa=2.0)

    pf = ParticleFilter(resampler=SoftResample(alpha=1.0), ssm=ungm, n_particles=n_particles)

    X_kf, X_ekf, X_ukf, X_pf = [], [], [], []

    x_curr_kf, P_curr_kf = x0, P0
    x_curr_ekf, P_curr_ekf = x0, P0
    x_curr_ukf, P_curr_ukf = x0, P0

    # PF State initialized [Batch=1, n_particles, Dim=1]
    p_curr_pf = tf.random.normal([1, n_particles, 1], mean=x0[0], stddev=tf.sqrt(P0[0, 0]))
    w_curr_pf = tf.ones([1, n_particles], dtype=tf.float32) / float(n_particles)

    # 3. Tracking Loops
    print("Tracking in progress...")
    start_time = time.perf_counter()

    for n in range(1, n_steps + 1):
        step_tensor = tf.constant(n, dtype=tf.float32)
        y_curr = Y_obs[n - 1]

        # --- Naive KF ---
        x_pred_kf, P_pred_kf = kf.predict(x_curr_kf, P_curr_kf, A_naive, ungm.Q)
        x_curr_kf, P_curr_kf = kf.update(x_pred_kf, P_pred_kf, y_curr, C_naive, ungm.R)
        X_kf.append(x_curr_kf)

        # --- EKF ---
        x_pred_ekf, P_pred_ekf = ekf.predict(x_curr_ekf, P_curr_ekf, t=step_tensor)
        x_curr_ekf, P_curr_ekf = ekf.update(x_pred_ekf, P_pred_ekf, y_curr, t=step_tensor)
        X_ekf.append(x_curr_ekf)

        # --- UKF ---
        x_pred_ukf, P_pred_ukf, sigs = ukf.predict(x_curr_ukf, P_curr_ukf, t=step_tensor)
        x_curr_ukf, P_curr_ukf = ukf.update(x_pred_ukf, P_pred_ukf, sigs, y_curr, t=step_tensor)
        X_ukf.append(x_curr_ukf)

        # --- PF ---
        p_pred_pf = pf.predict(p_curr_pf, t=step_tensor)
        p_curr_pf, w_curr_pf, x_est_pf, ess = pf.update(p_pred_pf, w_curr_pf, y_curr, t=step_tensor)
        p_curr_pf, w_curr_pf = pf.resample_if_needed(p_curr_pf, w_curr_pf, ess)
        X_pf.append(tf.squeeze(x_est_pf))

    print(f"Tracking complete in {(time.perf_counter() - start_time):.2f} seconds.")

    # 4. Metrics and Visualization (Pure TensorFlow)
    X_true_tf = tf.reshape(X_true, [-1])
    X_kf_tf = tf.reshape(tf.stack(X_kf), [-1])
    X_ekf_tf = tf.reshape(tf.stack(X_ekf), [-1])
    X_ukf_tf = tf.reshape(tf.stack(X_ukf), [-1])
    X_pf_tf = tf.reshape(tf.stack(X_pf), [-1])

    rmse_kf = float(tf.sqrt(tf.reduce_mean(tf.square(X_true_tf - X_kf_tf))))
    rmse_ekf = float(tf.sqrt(tf.reduce_mean(tf.square(X_true_tf - X_ekf_tf))))
    rmse_ukf = float(tf.sqrt(tf.reduce_mean(tf.square(X_true_tf - X_ukf_tf))))
    rmse_pf = float(tf.sqrt(tf.reduce_mean(tf.square(X_true_tf - X_pf_tf))))

    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    axs[0].plot(X_true_tf, 'k-', linewidth=3, alpha=0.4, label='True State')
    axs[0].plot(X_kf_tf, 'r:', linewidth=1.5, alpha=0.5, label=f'Naive KF (RMSE: {rmse_kf:.1f})')
    axs[0].plot(X_ekf_tf, 'b--', linewidth=1.5, label=f'EKF (RMSE: {rmse_ekf:.2f})')
    axs[0].plot(X_ukf_tf, 'g-.', linewidth=1.5, label=f'UKF (RMSE: {rmse_ukf:.2f})')
    axs[0].plot(X_pf_tf, 'm-', linewidth=1.5, alpha=0.9, label=f'PF (RMSE: {rmse_pf:.2f})')

    axs[0].set_title('UNGM Tracking: Linear vs. Non-linear vs. Ensemble Methods')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    y_min = float(tf.reduce_min(X_true_tf)) - 20.0
    y_max = float(tf.reduce_max(X_true_tf)) + 20.0
    axs[0].set_ylim([y_min, y_max])

    axs[1].plot(tf.abs(X_true_tf - X_ekf_tf), 'b--', alpha=0.6, label='EKF Error')
    axs[1].plot(tf.abs(X_true_tf - X_ukf_tf), 'g-.', alpha=0.6, label='UKF Error')
    axs[1].plot(tf.abs(X_true_tf - X_pf_tf), 'm-', alpha=0.8, label='PF Error')

    axs[1].set_title('Estimation Error Comparison (KF omitted due to massive scale)')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Absolute Error')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_ungm_comparison(n_particles=200, n_steps=100, q_var=10.0, r_var=1.0)