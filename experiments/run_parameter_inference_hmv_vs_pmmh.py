import tensorflow as tf
import tensorflow_probability as tfp
import time

# --- Centralized Imports ---
from resamplers.dresamplers import OTResample
from dpf.differentiablePF import *
from benchmarks import UNGM_SSM
import visualization as viz


def run_parameter_inference(n_results=100, n_steps=40):
    tf.random.set_seed(42)

    # 1. Generate Data using established UNGM_SSM
    true_Q, true_R = 10.0, 1.0
    ssm_data = UNGM_SSM(sigma_v_sq=true_Q, sigma_w_sq=true_R)

    T_train, T_val = 40, 50
    _, obs_train = ssm_data.generate_data(T_train, tf.constant([0.1]))
    obs_tensor = tf.expand_dims(obs_train, 0)

    val_gt, val_obs = ssm_data.generate_data(T_val, tf.constant([0.1]))
    val_gt_tensor = tf.expand_dims(val_gt, 0)
    val_obs_tensor = tf.expand_dims(val_obs, 0)

    # 2. Setup Differentiable Filter
    ssm_filter = UNGM_SSM(sigma_v_sq=1.0, sigma_w_sq=1.0)  # Dummy Q/R, MCMC overrides them
    # filter_mod = Differentiable_Li17EDH_Filter(ssm_filter, OTResample(epsilon=0.1, n_iters=10))
    filter_mod = Differentiable_Li17_Filter(ssm_filter, OTResample(epsilon=0.1, n_iters=10))

    @tf.function
    def target_log_prob(log_Q, log_R):
        _, ll = filter_mod(obs_tensor, log_Q, log_R, N=30)
        prior = -0.5 * ((log_Q - tf.math.log(10.0)) ** 2 + (log_R - tf.math.log(1.0)) ** 2)
        return ll + prior

    init_state = [tf.constant(tf.math.log(5.0)), tf.constant(tf.math.log(2.0))]

    # 3. HMC (Gradient-Based)
    print(">>> Running HMC (Gradient-Based)...")
    t0 = time.time()
    hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(target_log_prob, step_size=0.05, num_leapfrog_steps=5),
        num_adaptation_steps=n_steps)

    hmc_samples, hmc_results = tfp.mcmc.sample_chain(
        num_results=n_results, num_burnin_steps=n_steps,
        current_state=init_state, kernel=hmc_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
    hmc_time = time.time() - t0

    # 4. PMMH (Random Walk)
    print(">>> Running PMMH (Random Walk)...")
    t0 = time.time()

    def rwm_proposal(state, seed):
        return tfp.mcmc.random_walk_normal_fn(scale=0.1)(state, seed)

    pmmh_kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob, new_state_fn=rwm_proposal)

    pmmh_samples, pmmh_results = tfp.mcmc.sample_chain(
        num_results=n_results, num_burnin_steps=40,
        current_state=init_state, kernel=pmmh_kernel,
        trace_fn=lambda _, pkr: pkr.is_accepted)
    pmmh_time = time.time() - t0

    # Extract Q samples
    hmc_Q_vals = tf.exp(hmc_samples[0]).numpy()
    pmmh_Q_vals = tf.exp(pmmh_samples[0]).numpy()

    # Extract R samples
    hmc_R_vals = tf.exp(hmc_samples[1]).numpy()
    pmmh_R_vals = tf.exp(pmmh_samples[1]).numpy()

    # Calculate Effective Sample Size (ESS) for both
    hmc_ess_Q = tf.reduce_mean(tfp.mcmc.effective_sample_size(hmc_samples[0])).numpy()
    hmc_ess_R = tf.reduce_mean(tfp.mcmc.effective_sample_size(hmc_samples[1])).numpy()
    pmmh_ess_Q = tf.reduce_mean(tfp.mcmc.effective_sample_size(pmmh_samples[0])).numpy()
    pmmh_ess_R = tf.reduce_mean(tfp.mcmc.effective_sample_size(pmmh_samples[1])).numpy()

    # Calculate Means for the final 20 samples to run the tracking validation
    hmc_lQ_mean = tf.reduce_mean(hmc_samples[0][-20:])
    hmc_lR_mean = tf.reduce_mean(hmc_samples[1][-20:])
    pmmh_lQ_mean = tf.reduce_mean(pmmh_samples[0][-20:])
    pmmh_lR_mean = tf.reduce_mean(pmmh_samples[1][-20:])

    # Run the filter one last time with the inferred parameters
    hmc_est, _ = filter_mod(val_obs_tensor, hmc_lQ_mean, hmc_lR_mean, N=200)
    pmmh_est, _ = filter_mod(val_obs_tensor, pmmh_lQ_mean, pmmh_lR_mean, N=200)

    hmc_rmse = float(tf.sqrt(tf.reduce_mean(tf.square(hmc_est - val_gt_tensor))))
    pmmh_rmse = float(tf.sqrt(tf.reduce_mean(tf.square(pmmh_est - val_gt_tensor))))

    # Print Full Joint Table
    print(f"\n{'Metric':<16} | {'HMC (Gradient)':<15} | {'PMMH (Random Walk)':<15}")
    print("-" * 54)
    print(f"{'Runtime (s)':<16} | {hmc_time:<15.4f} | {pmmh_time:<15.4f}")
    print(
        f"{'Accept Rate':<16} | {tf.reduce_mean(tf.cast(hmc_results, tf.float32)):<15.4f} | {tf.reduce_mean(tf.cast(pmmh_results, tf.float32)):<15.4f}")
    print(f"{'ESS (Q)':<16} | {hmc_ess_Q:<15.2f} | {pmmh_ess_Q:<15.2f}")
    print(f"{'ESS (R)':<16} | {hmc_ess_R:<15.2f} | {pmmh_ess_R:<15.2f}")
    print(f"{'RMSE Tracking':<16} | {hmc_rmse:<15.4f} | {pmmh_rmse:<15.4f}")

    viz.show_mcmc_summary_joint(hmc_Q_vals, pmmh_Q_vals, hmc_R_vals, pmmh_R_vals, val_gt_tensor, hmc_est, pmmh_est,
                                hmc_rmse, pmmh_rmse)

if __name__ == "__main__":
    run_parameter_inference(n_results=100, n_steps=40)
