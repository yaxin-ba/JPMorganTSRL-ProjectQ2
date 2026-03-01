import tensorflow as tf
import matplotlib.pyplot as plt

from utils.tune_parameters import tune_ot_parameters
from benchmarks import UNGM_SSM
from resamplers.dresamplers import SoftResample, OTResample
from dpf.differentiablePF import DifferentiableParticleFilter


def run_ot_soft_nonlinear_compare(n_particles=100, n_steps=50):
    tune_ot_parameters()

    print("=== MAIN EXPERIMENT: UNGM BENCHMARK ===")
    tf.random.set_seed(42)
    # T = 50

    # Generate data using your exact UNGM_SSM model
    # ssm = UNGM_SSM()
    ssm = UNGM_SSM(sigma_v_sq=1.0, sigma_w_sq=1.0)
    true_states, obs = ssm.generate_data(N=n_steps, x0=tf.constant([0.0]))

    # Inject Batch Dimension [B=1, T, D=1] for the DPF framework
    obs_batch = tf.expand_dims(obs, 0)
    true_states_batch = tf.expand_dims(true_states, 0)

    # n_particles = 100
    models = {
        'Soft-DPF': DifferentiableParticleFilter(ssm, SoftResample(alpha=0.5), n_particles=n_particles),
        'OT-DPF (Eps=0.1)': DifferentiableParticleFilter(ssm, OTResample(epsilon=0.1, n_iters=15), n_particles=n_particles)
    }

    results = {}

    for name, model in models.items():
        # Trace gradients securely backwards out of the generic tf.while_loop wrapper
        with tf.GradientTape() as tape:
            tape.watch(obs_batch)
            est, ess = model(obs_batch)
            loss = tf.reduce_mean(tf.square(est - true_states_batch))

        grads = tape.gradient(loss, obs_batch)
        grad_norm = tf.linalg.norm(grads)

        results[name] = {
            'loss': loss,
            'grad_norm': grad_norm,
            'est': tf.squeeze(est),
            'ess': tf.reduce_mean(ess, axis=0)
        }

    print(f"{'Method':<20} | {'RMSE':<10} | {'Grad Norm (Obs)':<15}")
    print("-" * 50)
    for name, res in results.items():
        rmse = tf.sqrt(res['loss'])
        print(f"{name:<20} | {rmse:<10.4f} | {res['grad_norm']:<15.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(tf.squeeze(true_states_batch).numpy(), 'k-', lw=2, label='True State')
    for name, res in results.items():
        axs[0].plot(res['est'].numpy(), '--', label=name)
    axs[0].set_title('UNGM Tracking (Differentiable PFs)')
    axs[0].legend()

    for name, res in results.items():
        axs[1].plot(res['ess'].numpy(), label=name)
    axs[1].set_title('Effective Sample Size (ESS)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_ot_soft_nonlinear_compare(n_particles=100, n_steps=50)