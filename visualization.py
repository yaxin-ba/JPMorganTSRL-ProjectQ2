import matplotlib.pyplot as plt
import tensorflow as tf


def plot_advanced_metrics(prob_name, true_states, results_dict, profile_data):
    """
    3-Panel Dashboard:
    1. Trajectory Plot
    2. RMSE vs Latency (Speed/Accuracy Trade-off)
    3. Storage (Parameter Count)
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Trajectory (Same as before)
    true_np = tf.squeeze(true_states).numpy()
    axs[0].plot(true_np, 'k-', label='Truth', linewidth=2)
    for name, est in results_dict.items():
        axs[0].plot(tf.squeeze(est).numpy(), '--', label=name, alpha=0.7)
    axs[0].set_title(f"{prob_name}: Trajectory")
    axs[0].legend(fontsize='x-small')

    # Panel 2: Speed vs. Accuracy (Scatter Plot)
    #
    for name in results_dict.keys():
        rmse = tf.sqrt(tf.reduce_mean(tf.square(results_dict[name] - true_states))).numpy()
        latency = profile_data[name]['latency']
        axs[1].scatter(latency, rmse, s=100, label=name)
        axs[1].annotate(name, (latency, rmse), fontsize=8)

    axs[1].set_xlabel("Latency (ms) - Lower is Better")
    axs[1].set_ylabel("RMSE - Lower is Better")
    axs[1].set_title("Speed vs. Accuracy Trade-off")
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Panel 3: Model Storage (Bar Chart)
    names = list(profile_data.keys())
    sizes = [data['size'].numpy() for data in profile_data.values()]
    axs[2].bar(names, sizes, color='plum')
    axs[2].set_title("Model Complexity (Trainable Params)")
    axs[2].set_ylabel("Count")
    axs[2].tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    plt.show()


def show_benchmark_summary(prob_name, true_states, results, stats):
    """A comprehensive 3-panel view: Trajectory, Efficiency, and Gradient Health."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Trajectory Tracking
    # Squeeze removes batch dimensions, allowing matplotlib to plot the TF tensor directly
    true_arr = tf.squeeze(true_states)
    ax[0].plot(true_arr, 'k-', label='Ground Truth', alpha=0.6, linewidth=2)

    for name, est in results.items():
        est_arr = tf.squeeze(est)
        ax[0].plot(est_arr, '--', label=name, alpha=0.8)

    ax[0].set_title(f"{prob_name}: State Trajectory")
    ax[0].legend(fontsize='x-small', loc='upper right')
    ax[0].grid(True, alpha=0.3)

    # 2. Efficiency Frontier (RMSE vs Speed)
    for name in results.keys():
        rmse = stats[name]['rmse']
        ms = stats[name]['latency']
        ax[1].scatter(ms, rmse, s=120, edgecolors='black', alpha=0.7)
        ax[1].annotate(name, (ms, rmse), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax[1].set_xlabel("Latency (ms)")
    ax[1].set_ylabel("RMSE (Accuracy)")
    ax[1].set_title("Speed vs. Accuracy Trade-off")
    ax[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Gradient Stability (Log Scale)
    names = list(results.keys())
    norms = [stats[n]['grad_norm'] for n in names]

    # Color active gradients teal, dead gradients (0.0) gray
    colors = ['teal' if n > 0 else 'gray' for n in norms]

    # Add a tiny epsilon to prevent log(0) errors in the plot rendering
    safe_norms = [n + 1e-12 for n in norms]

    ax[2].bar(names, safe_norms, color=colors, edgecolor='black', alpha=0.8)
    ax[2].set_yscale('log')
    ax[2].set_title("Gradient Norm (Differentiability)")
    ax[2].set_ylabel("Log Norm Magnitude")
    ax[2].tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import tensorflow as tf

def show_mcmc_summary_joint(hmc_Q, pmmh_Q, hmc_R, pmmh_R, val_gt, hmc_est, pmmh_est, hmc_rmse, pmmh_rmse):
    """Plots joint traces and posteriors for both Q and R, plus tracking."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # --- Parameter Q ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hmc_Q, label='HMC', alpha=0.8)
    ax1.plot(pmmh_Q, label='PMMH', alpha=0.6)
    ax1.axhline(10.0, c='r', ls='--', label='True Q')
    ax1.set_title('MCMC Trace (Q)')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(hmc_Q, alpha=0.5, bins=15, label='HMC')
    ax2.hist(pmmh_Q, alpha=0.5, bins=15, label='PMMH')
    ax2.axvline(10.0, c='r', ls='--')
    ax2.set_title('Posterior Density (Q)')

    # --- Parameter R ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(hmc_R, label='HMC', alpha=0.8)
    ax3.plot(pmmh_R, label='PMMH', alpha=0.6)
    ax3.axhline(1.0, c='r', ls='--', label='True R')
    ax3.set_title('MCMC Trace (R)')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(hmc_R, alpha=0.5, bins=15, label='HMC')
    ax4.hist(pmmh_R, alpha=0.5, bins=15, label='PMMH')
    ax4.axvline(1.0, c='r', ls='--')
    ax4.set_title('Posterior Density (R)')

    # --- State Tracking ---
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.plot(tf.squeeze(val_gt), 'k-', label='True State', linewidth=2)
    ax5.plot(tf.squeeze(hmc_est), 'b--', label=f'HMC (RMSE={hmc_rmse:.2f})')
    ax5.plot(tf.squeeze(pmmh_est), 'orange', ls=':', label=f'PMMH (RMSE={pmmh_rmse:.2f})')
    ax5.set_title('Validation Tracking (Inferred Parameters)')
    ax5.legend()

    plt.tight_layout()
    plt.show()