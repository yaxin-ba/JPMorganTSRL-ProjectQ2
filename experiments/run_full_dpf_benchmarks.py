import tensorflow as tf
import os

# Core logic
from resamplers.resamplers import MultinomialResample, SystematicResampler
from resamplers.dresamplers import SoftResample, OTResample
from utils.tune_parameters import tune_ot_parameters
from utils.training import train_model, train_deeponet_ot, train_gradnet_ot

# Models & Filters
from benchmarks import LinearGaussianSSM, UNGM_SSM
from models.layers import ConvexDeepONet, ICNN, OT_TransportMap
from dpf.differentiablePF import DifferentiableParticleFilter
from dpf.neural_filters import NeuralProposal_Filter, NF_DPF_Filter, NeuralOT_Resampler_Filter, SIS_Filter

# Metrics & Visualization
from metrics.accuracy import calculate_rmse
from metrics.system import measure_latency, get_parameter_count
from metrics.optimization import get_grad_norm
import visualization as viz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_full_benchmark(n_particles=50, n_steps=50):
    # Strictly using TensorFlow's native random seed
    tf.random.set_seed(42)

    ot_eps, ot_iters = tune_ot_parameters()

    class LGSSM_Wrapper(LinearGaussianSSM):
        @tf.function
        def f_fn(self, x, t=None): return tf.linalg.matvec(self.A, x)

        @tf.function
        def h_fn(self, x, t=None): return tf.linalg.matvec(self.C, x)

    problems = [
        ("LGSSM", LGSSM_Wrapper(A=[[0.5]], B=[[0.1]], C=[[1.0]], D=[[0.1]])),
        ("UNGM", UNGM_SSM(sigma_v_sq=1.0, sigma_w_sq=1.0))
    ]

    for prob_name, ssm in problems:
        # T = 50
        true_states, obs_list = ssm.generate_data(n_steps, tf.constant([0.0]))
        true_states_batch = tf.expand_dims(true_states, 0)
        obs_batch = tf.expand_dims(obs_list, 0)
        # N_part = 100

        models = {
            'PF': DifferentiableParticleFilter(ssm, MultinomialResample(), n_particles=n_particles),
            'Systematic PF': DifferentiableParticleFilter(ssm, SystematicResampler(), n_particles=n_particles),
            'SIS': SIS_Filter(ssm, n_particles=n_particles),
            'SPF': DifferentiableParticleFilter(ssm, SoftResample(0.5), n_particles=n_particles),
            'OT-DPF': DifferentiableParticleFilter(ssm, OTResample(ot_eps, ot_iters), n_particles=n_particles),
            'Neural Proposal': NeuralProposal_Filter(ssm, SoftResample(0.5), n_particles=n_particles),
            'Neural-OT': NeuralProposal_Filter(ssm, OTResample(ot_eps, ot_iters), n_particles=n_particles),
            'NF-Flow-OT': NF_DPF_Filter(ssm, SoftResample(0.5), n_particles=n_particles),
            # NeuralOT_Resampler_Filter completely replaces discrete resampling
            'DeepONet-OT': NeuralOT_Resampler_Filter(ssm, OT_TransportMap(ConvexDeepONet(1, 2)), n_particles=n_particles),
            'GradNet-OT': NeuralOT_Resampler_Filter(ssm, OT_TransportMap(ICNN(1, 2)), n_particles=n_particles)
        }

        print(f"\n--- Running Benchmark: {prob_name} ---")
        print(f"{'Method':<16} | {'RMSE':<10} | {'Grad Norm':<10} | {'Latency (ms)':<12}")
        print("-" * 55)

        results = {}
        stats = {}

        for name, model in models.items():
            # Strict Neural Routing
            if name == 'DeepONet-OT':
                print(f" > Training {name} with PINO Loss...")
                model = train_deeponet_ot(model, obs_batch, true_states_batch, epochs=300)
            elif name == 'GradNet-OT':
                print(f" > Training {name} with Monge-Ampère Loss...")
                model = train_gradnet_ot(model, obs_batch, true_states_batch, epochs=300)
            elif "Neural" in name or "NF" in name:
                print(f" > Training {name}...")
                model = train_model(model, obs_batch, true_states_batch, epochs=150)

            # Metric Collection
            with tf.GradientTape() as tape:
                tape.watch(obs_batch)
                est, _ = model(obs_batch)
                rmse_val = calculate_rmse(est, true_states_batch)

            # Converting TF Tensors to pure Python primitives
            stats[name] = {
                'rmse': float(rmse_val),
                'latency': float(measure_latency(model, obs_batch)),
                'grad_norm': float(get_grad_norm(tape, est, obs_batch)),
                'params': int(get_parameter_count(model))
            }
            results[name] = est

            print(
                f"{name:<16} | {stats[name]['rmse']:<10.4f} | {stats[name]['grad_norm']:<10.4f} | {stats[name]['latency']:<12.2f}")

        # Trigger Dashboard Render
        viz.show_benchmark_summary(prob_name, true_states_batch, results, stats)


if __name__ == "__main__":
    run_full_benchmark(n_particles=50, n_steps=50)