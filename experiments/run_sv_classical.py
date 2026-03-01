import tensorflow as tf
import time
# import numpy as np
import matplotlib.pyplot as plt

from benchmarks import StochasticVolatilitySSM
from classical_filters.kalman_filters import ExtendedKalmanFilter, UnscentedKalmanFilter


def run_sv_classical(n_steps=200):
    """
    Executes a tracking comparison between EKF and UKF for a Stochastic Volatility model.

    This function generates synthetic volatility data, performs inference using
    classical non-linear filters, and visualizes the estimation error and RMSE.

    Args:
        n_steps (int): Total number of simulation time steps. Defaults to 200.
    """
    print("--- Running TF-Native SV Model Comparison ---")

    # 1. Setup Data
    tf.random.set_seed(42)
    # n_steps = 200

    # Init SV Model (alpha=0.91, sigma=1.0, beta=0.5)
    sv_model = StochasticVolatilitySSM()

    # Determine theoretical initial variance: sigma^2 / (1 - alpha^2)
    init_variance = (sv_model.sigma ** 2) / (1.0 - sv_model.alpha ** 2)
    x0 = tf.constant([0.0], dtype=tf.float32)

    # Generate ground truth and transformed observations Z = Y^2
    X_true, Z_obs = sv_model.generate_data(n_steps, x0)

    # 2. Filter Initialization
    P0 = tf.reshape(init_variance, [1, 1])
    R_dyn_base = tf.constant([[1e-6]], dtype=tf.float32)  # Small base to prevent singular matrices

    ekf = ExtendedKalmanFilter(ssm=sv_model)
    ukf = UnscentedKalmanFilter(ssm=sv_model)

    X_ekf, X_ukf = [], []

    x_curr_ekf, P_curr_ekf = x0, P0
    x_curr_ukf, P_curr_ukf = x0, P0

    print("Tracking in progress...")
    start_time = time.perf_counter()

    for n in range(n_steps):
        z_curr = Z_obs[n]

        # --- EKF Loop ---
        x_pred_ekf, P_pred_ekf = ekf.predict(x_curr_ekf, P_curr_ekf)

        # Dynamic R for SV model Z transform: R_dyn = 2 * (Expected Z)^2
        h_x_ekf = sv_model.h_fn(x_pred_ekf)
        R_dyn_ekf = 2.0 * tf.square(h_x_ekf) + R_dyn_base

        x_curr_ekf, P_curr_ekf = ekf.update(x_pred_ekf, P_pred_ekf, z_curr, R_dyn=R_dyn_ekf)
        X_ekf.append(x_curr_ekf)

        # --- UKF Loop ---
        x_pred_ukf, P_pred_ukf, sigs = ukf.predict(x_curr_ukf, P_curr_ukf)

        h_x_ukf = sv_model.h_fn(x_pred_ukf)
        R_dyn_ukf = 2.0 * tf.square(h_x_ukf) + R_dyn_base

        x_curr_ukf, P_curr_ukf = ukf.update(x_pred_ukf, P_pred_ukf, sigs, z_curr, R_dyn=R_dyn_ukf)
        X_ukf.append(x_curr_ukf)

    print(f"Tracking complete in {(time.perf_counter() - start_time):.2f} seconds.")

    # # 3. Visualization
    # X_true_np = X_true.numpy().flatten()
    # X_ekf_np = tf.stack(X_ekf).numpy().flatten()
    # X_ukf_np = tf.stack(X_ukf).numpy().flatten()
    #
    # rmse_ekf = np.sqrt(np.mean((X_true_np - X_ekf_np) ** 2))
    # rmse_ukf = np.sqrt(np.mean((X_true_np - X_ukf_np) ** 2))
    #
    # fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    #
    # axs[0].plot(X_true_np, 'k-', alpha=0.5, label='True Volatility')
    # axs[0].plot(X_ekf_np, 'b--', label=f'EKF (RMSE: {rmse_ekf:.3f})')
    # axs[0].plot(X_ukf_np, 'g-', alpha=0.8, label=f'UKF (RMSE: {rmse_ukf:.3f})')
    # axs[0].set_title('Volatility Tracking Performance (Transformed Obs $Y^2$)')
    # axs[0].legend()
    # axs[0].grid(True)
    #
    # axs[1].plot(np.abs(X_true_np - X_ekf_np), 'b-', alpha=0.5, label='EKF Error')
    # axs[1].plot(np.abs(X_true_np - X_ukf_np), 'g-', alpha=0.5, label='UKF Error')
    # axs[1].set_title('Absolute Estimation Error')
    # axs[1].set_xlabel('Time Step')
    # axs[1].set_ylabel('Error')
    # axs[1].legend()
    # axs[1].grid(True)
    #
    # plt.tight_layout()
    # plt.show()
    # 3. Visualization (Pure TensorFlow & Matplotlib)
    X_true_tf = tf.reshape(X_true, [-1])
    X_ekf_tf = tf.reshape(tf.stack(X_ekf), [-1])
    X_ukf_tf = tf.reshape(tf.stack(X_ukf), [-1])

    # Pure TF RMSE calculation
    rmse_ekf = float(tf.sqrt(tf.reduce_mean(tf.square(X_true_tf - X_ekf_tf))))
    rmse_ukf = float(tf.sqrt(tf.reduce_mean(tf.square(X_true_tf - X_ukf_tf))))

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    axs[0].plot(X_true_tf, 'k-', alpha=0.5, label='True Volatility')
    axs[0].plot(X_ekf_tf, 'b--', label=f'EKF (RMSE: {rmse_ekf:.3f})')
    axs[0].plot(X_ukf_tf, 'g-', alpha=0.8, label=f'UKF (RMSE: {rmse_ukf:.3f})')
    axs[0].set_title('Volatility Tracking Performance (Transformed Obs $Y^2$)')
    axs[0].legend()
    axs[0].grid(True)

    # Pure TF Absolute Error calculation for plotting
    axs[1].plot(tf.abs(X_true_tf - X_ekf_tf), 'b-', alpha=0.5, label='EKF Error')
    axs[1].plot(tf.abs(X_true_tf - X_ukf_tf), 'g-', alpha=0.5, label='UKF Error')
    axs[1].set_title('Absolute Estimation Error')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Error')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_sv_classical(n_steps=200)