# import argparse
# import tensorflow as tf
# import numpy as np
#
# from experiments.run_acoustic_tracking import run_acoustic_tracking
# from experiments.run_full_dpf_benchmarks import run_full_benchmark
# from experiments.run_kernel_flow import run_kernel_flow
# from experiments.run_stochasticPF_OC import run_tracking_scenario
# from experiments.run_parameter_inference_hmv_vs_pmmh import run_parameter_inference
# from experiments.run_sv_classical import run_sv_classical
# from experiments.run_ungm_classical import run_ungm_comparison
# from experiments.run_ungm_dpf_compare_ot_soft import run_ot_soft_compare
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Centralized Runner for DPF and Classical Filter Experiments")
#
#     # Experiment Router
#
#     parser.add_argument('--exp', type=str, required=True,
#                         choices=[
#                             'acoustic', 'full_dpf', 'kernel', 'stiffness_migrate',
#                             'pmcmc', 'sv_class', 'ungm_class', 'ungm_dpf'
#                         ],
#                         help="Which experiment to run.")
#
#     # Global Hyperparameters
#     parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
#     parser.add_argument('--n_particles', type=int, default=100, help="Number of particles for the filters.")
#     parser.add_argument('--n_steps', type=int, default=50, help="Number of time steps to simulate.")
#
#     # Model Specific Tuning Parameters (Add more as needed)
#     parser.add_argument('--q_var', type=float, default=10.0, help="Process noise variance (Q).")
#     parser.add_argument('--r_var', type=float, default=1.0, help="Observation noise variance (R).")
#     parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for Neural filters.")
#     parser.add_argument('--n_results', type=int, default=100, help="Number of results.")
#
#     return parser.parse_args()
#
#
# def set_seeds(seed):
#     """Ensure reproducibility across all libraries."""
#     tf.random.set_seed(seed)
#     np.random.seed(seed)
#     print(f"Global seed set to: {seed}")
#
#
# def main():
#     args = parse_args()
#     set_seeds(args.seed)
#
#     print(f"=== Starting Experiment: {args.exp.upper()} ===")
#     print(f"Parameters: N_particles={args.n_particles}, N_steps={args.n_steps}, Q={args.q_var}, R={args.r_var}\n")
#
#     # Route to the correct experiment function and pass the parsed arguments
#     if args.exp == 'acoustic':
#         run_acoustic_tracking(n_particles=args.n_particles, n_steps=args.n_steps)
#     elif args.exp == 'full_dpf':
#         run_full_benchmark(n_particles=args.n_particles, n_steps=args.n_steps)
#     elif args.exp == 'kernel':
#         run_kernel_flow(n_particles=args.n_particles)
#     elif args.exp == 'stiffness_migrate':
#         run_tracking_scenario(n_steps=args.n_steps)
#     elif args.exp == 'pmcmc':
#         run_parameter_inference(n_results=args.n_results, n_steps=args.n_steps)
#     elif args.exp == 'sv_class':
#         run_sv_classical(n_steps=args.n_steps)
#     elif args.exp == 'ungm_class':
#         run_ungm_comparison(n_particles=args.n_particles, n_steps=args.n_steps, q_var=args.q_var, r_var=args.r_var)
#     elif args.exp == 'ungm_dpf':
#         run_ot_soft_compare(n_particles=args.n_particles, n_steps=args.n_steps)
#
#
# if __name__ == "__main__":
#     main()


"""
Main entry point for the From Particle Filter to Particle Flow framework.

This script acts as a centralized runner for various experiments, including
classical filtering comparisons, particle filters and particle flows benchmarks,
differentiable particle filters, and Bayesian parameter inference.
"""

import argparse
import tensorflow as tf

from experiments.run_acoustic_tracking import run_acoustic_tracking
from experiments.run_full_dpf_benchmarks import run_full_benchmark
from experiments.run_kernel_flow import run_kernel_flow
from experiments.run_stochasticPF_OC import run_tracking_scenario
from experiments.run_parameter_inference_hmv_vs_pmmh import run_parameter_inference
from experiments.run_sv_classical import run_sv_classical
from experiments.run_ungm_classical import run_ungm_comparison
from experiments.run_ungm_dpf_compare_ot_soft import run_ot_soft_nonlinear_compare


def parse_args():
    """
    Parses command-line arguments for experiment routing and hyperparameter tuning.

    Returns:
        argparse.Namespace: A namespace object containing parsed experimental
        parameters and configurations.
    """
    parser = argparse.ArgumentParser(description="Centralized Runner for DPF and Classical Filter Experiments")

    # Experiment Router
    parser.add_argument('--exp', type=str, required=True,
                        choices=[
                            'acoustic', 'full_dpf', 'kernel', 'stiffness_migrate',
                            'pmcmc', 'sv_class', 'ungm_class', 'ungm_dpf'
                        ],
                        help="The keyword for the specific experiment to execute.")

    # Global Hyperparameters
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--n_particles', type=int, default=100, help="Number of particles for filter-based methods.")
    parser.add_argument('--n_steps', type=int, default=50, help="Total number of simulation time steps.")

    # Model Specific Tuning Parameters
    parser.add_argument('--q_var', type=float, default=10.0, help="Process noise variance (Q).")
    parser.add_argument('--r_var', type=float, default=1.0, help="Observation noise variance (R).")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for gradient-based updates.")
    parser.add_argument('--n_results', type=int, default=100, help="Number of samples to draw in MCMC/PMCMC chains.")

    return parser.parse_args()


def set_seeds(seed):
    """
    Sets global seeds for TensorFlow to ensure deterministic results.

    Args:
        seed (int): The integer seed value.
    """
    tf.random.set_seed(seed)
    print(f"Global seed set to: {seed}")


def main():
    """
    Routes execution to the selected experiment based on parsed CLI arguments.
    Initializes global settings and prints configuration summaries before runtime.
    """
    args = parse_args()
    set_seeds(args.seed)

    print(f"=== Starting Experiment: {args.exp.upper()} ===")
    print(f"Parameters: N_particles={args.n_particles}, N_steps={args.n_steps}, Q={args.q_var}, R={args.r_var}\n")

    # Route to the correct experiment function and pass the parsed arguments
    if args.exp == 'acoustic':
        run_acoustic_tracking(n_particles=args.n_particles, n_steps=args.n_steps)
    elif args.exp == 'full_dpf':
        run_full_benchmark(n_particles=args.n_particles, n_steps=args.n_steps)
    elif args.exp == 'kernel':
        run_kernel_flow(n_particles=args.n_particles)
    elif args.exp == 'stiffness_migrate':
        run_tracking_scenario(n_steps=args.n_steps)
    elif args.exp == 'pmcmc':
        run_parameter_inference(n_results=args.n_results, n_steps=args.n_steps)
    elif args.exp == 'sv_class':
        run_sv_classical(n_steps=args.n_steps)
    elif args.exp == 'ungm_class':
        run_ungm_comparison(n_particles=args.n_particles, n_steps=args.n_steps, q_var=args.q_var, r_var=args.r_var)
    elif args.exp == 'ungm_dpf':
        run_ot_soft_nonlinear_compare(n_particles=args.n_particles, n_steps=args.n_steps)


if __name__ == "__main__":
    main()