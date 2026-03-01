particle_inference/
├── core/                   # Base abstract classes and interfaces
│   ├── __init__.py
│   ├── ssm.py              # StateSpaceModel base class
│   ├── filter.py           # AbstractFilter base class
│   └── resampler.py        # AbstractResampler base class
├── models/                 # Instantiations of SSMs
│   ├── __init__.py
│   ├── linear_gaussian.py
│   ├── nonlinear_benchmarks.py  # e.g., Lorenz-63, Pendulum
│   └── deep_models.py      # DeepONet, LSTM state transitions
├── filters/                # Classical filters
│   ├── __init__.py
│   ├── kalman.py           # KF, EKF, UKF
│   └── particle.py         # Standard PF (Bootstrap, SIR)
├── flows/                  # Particle Flow filters
│   ├── __init__.py
│   ├── edh.py              # Exact Daum and Huang
│   ├── ledh.py             # Local Exact Daum and Huang
│   ├── kernel_flow.py      # Kernel-based PF
│   └── stochastic_flow.py  # Optimal control / Homotopy flow
├── dpf/                    # Differentiable Particle Filters
│   ├── __init__.py
│   ├── soft_resample.py    # Softmax/relaxed resampling
│   └── ot_resample.py      # Optimal Transport (Sinkhorn) resampling
├── advanced/               # High-level sampling & Flow models
│   ├── __init__.py
│   ├── hmc.py              # Hamiltonian Monte Carlo integration
│   ├── normalizing_flow.py # NF for proposal generation
│   └── neural_sampler.py   # Neural Accelerating Sampling
├── utils/                  # Math, configs, and metrics
│   ├── __init__.py
│   ├── metrics.py          # ESS, RMSE, NLL, Gradient variance
│   ├── math_ops.py         # Custom TF Jacobians, Hessians
│   └── config.py           # Reproducibility constraints
├── experiments/            # Run scripts and plotting
│   ├── run_comparison.py
│   └── visualize.py
└── tests/                  # PyTest suite
    ├── test_ssm.py
    ├── test_filters.py
    └── test_gradients.py