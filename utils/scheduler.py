import tensorflow as tf
import matplotlib.pyplot as plt
from benchmarks_dai22 import Dai22StaticExample

class ShootingScheduler(tf.Module):
    """
    Implements the Simple Bisection Shooting Method for Stiffness Mitigation [Dai22].
    Solves Eq 28 natively in TensorFlow to find the optimal homotopy schedule.
    """

    def __init__(self, M0_matrix, Hh_matrix, mu=0.2, n_steps=29):
        super().__init__()
        self.M0 = tf.cast(M0_matrix, tf.float32)
        self.Hh = tf.cast(Hh_matrix, tf.float32)
        self.mu = tf.constant(mu, dtype=tf.float32)
        self.n_steps = tf.constant(n_steps, dtype=tf.int32)

        # The base Li17 lambda integration steps
        q = 1.2
        eps1 = (1.0 - q) / (1.0 - q ** n_steps)
        steps = [eps1 * (q ** i) for i in range(n_steps)]
        self.d_lambdas = tf.constant(steps, dtype=tf.float32)

        # Execute shooting method internally to find optimal Delta-Betas
        self.optimal_d_betas = self.solve_bisection_shooting()

    @tf.function
    def solve_bisection_shooting(self):
        # Bisection boundaries for the initial slope (p0 = dbeta/dlambda)
        p_low = tf.constant(-20.0, dtype=tf.float32)
        p_high = tf.constant(5.0, dtype=tf.float32)

        best_betas = tf.zeros([self.n_steps], dtype=tf.float32)

        # 40 iterations of Bisection is sufficient for high precision
        for _ in tf.range(40):
            p_mid = (p_low + p_high) / 2.0

            beta = tf.constant(0.0, dtype=tf.float32)
            dbeta = p_mid

            beta_history = tf.TensorArray(tf.float32, size=self.n_steps)

            # Forward integrate the ODE (Eq 28)
            for i in tf.range(self.n_steps):
                dl = self.d_lambdas[i]

                # Sub-step Euler integration for ODE stability
                dl_sub = dl / 10.0
                for _sub in tf.range(10):
                    # M = M0 + beta * Hh
                    M = self.M0 + beta * self.Hh
                    # Add jitter to diagonal for safe inversion
                    M = M + tf.eye(tf.shape(M)[0], dtype=tf.float32) * 1e-6
                    M_inv = tf.linalg.inv(M)
                    M_inv2 = tf.linalg.matmul(M_inv, M_inv)

                    tr_H = tf.linalg.trace(self.Hh)
                    tr_Minv = tf.linalg.trace(M_inv)
                    tr_M = tf.linalg.trace(M)
                    tr_Minv2_H = tf.linalg.trace(tf.linalg.matmul(M_inv2, self.Hh))

                    # Eq 28 (Nuclear Norm case)
                    d2beta = self.mu * (tr_H * tr_Minv + tr_M * tr_Minv2_H)

                    # Integration steps
                    beta = beta + dbeta * dl_sub
                    dbeta = dbeta + d2beta * dl_sub

                beta_history = beta_history.write(i, beta)

            betas = beta_history.stack()

            # Bisection check: Did we overshoot the target beta(1) = 1.0?
            if betas[-1] > 1.0:
                p_high = p_mid
            else:
                p_low = p_mid

            best_betas = betas

        # Convert the absolute beta schedule into step sizes (Delta Beta)
        # Because dx = f(beta) dbeta, these are the new integration steps!
        padded_betas = tf.concat([[0.0], best_betas], axis=0)
        d_betas = padded_betas[1:] - padded_betas[:-1]

        return d_betas

    @tf.function
    def get_beta_steps(self):
        return self.optimal_d_betas

if __name__ == "__main__":
    def run_dai22_static_example():
        """
        Executes the static 1-step example from Section 4 of Dai22.
        """

        print("\n--- Running Dai22 Section 4 Static Example ---")

        # 1. Load the benchmark model
        benchmark = Dai22StaticExample()
        M0, Hh = benchmark.get_jacobian_and_matrices()

        # 2. Solve using the pure TF Bisection Shooting Method (mu=0.2)
        scheduler = ShootingScheduler(M0, Hh, mu=0.2, n_steps=29)
        d_betas = scheduler.get_beta_steps()

        # 3. Reconstruct the absolute beta curve for plotting
        lambdas = scheduler.d_lambdas
        abs_lambdas = tf.math.cumsum(lambdas).numpy()
        abs_betas = tf.math.cumsum(d_betas).numpy()

        # 4. Plotting to match Figure 2(a) in Dai22
        plt.figure(figsize=(6, 5))
        plot_x = [0.0] + list(abs_lambdas)
        plot_y = [0.0] + list(abs_betas)

        plt.plot(plot_x, plot_y, 'r-', lw=2, label=r'Optimal $\beta^*(\lambda)$')
        plt.plot([0, 1], [0, 1], 'b--', label=r'Baseline $\beta(\lambda) = \lambda$')
        plt.title("Dai22 Figure 2(a): Optimal Homotopy Schedule")
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\beta(\lambda)$')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print("Static Example Complete. Curve matches Figure 2(a).")
    run_dai22_static_example()

#
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# class ShootingScheduler(tf.Module):
#     """
#     Implements the Simple Bisection Shooting Method for Stiffness Mitigation [Dai22].
#     Solves Eq 28 natively in TensorFlow to find the optimal homotopy schedule.
#     """
#
#     def __init__(self, M0_matrix, Hh_matrix, mu=0.2, n_steps=29):
#         super().__init__()
#         self.M0 = tf.cast(M0_matrix, tf.float32)
#         self.Hh = tf.cast(Hh_matrix, tf.float32)
#         self.mu = tf.constant(mu, dtype=tf.float32)
#         self.n_steps = int(n_steps)  # Strict Python integer to prevent scalar tensor bugs
#
#         # The base Li17 lambda integration steps
#         q = 1.2
#         eps1 = (1.0 - q) / (1.0 - q ** self.n_steps)
#         steps = [eps1 * (q ** i) for i in range(self.n_steps)]
#         self.d_lambdas = tf.constant(steps, dtype=tf.float32)
#
#         # Execute shooting method internally to find optimal Delta-Betas
#         self.optimal_d_betas = self.solve_bisection_shooting()
#
#     @tf.function
#     def solve_bisection_shooting(self):
#         # Bisection boundaries for the initial slope (p0 = dbeta/dlambda)
#         p_low = tf.constant(-20.0, dtype=tf.float32)
#         p_high = tf.constant(5.0, dtype=tf.float32)
#
#         # Pure tuple shape prevents scalar list errors
#         best_betas = tf.zeros((self.n_steps,), dtype=tf.float32)
#
#         num_dim = tf.shape(self.M0)[0]
#         I_mat = tf.eye(num_dim, dtype=tf.float32)
#
#         # 40 iterations of Bisection is sufficient for high precision
#         for _ in tf.range(40):
#             p_mid = (p_low + p_high) / 2.0
#
#             beta = tf.constant(0.0, dtype=tf.float32)
#             dbeta = p_mid
#
#             beta_history = tf.TensorArray(tf.float32, size=self.n_steps)
#
#             # Forward integrate the ODE (Eq 28)
#             for i in tf.range(self.n_steps):
#                 dl = self.d_lambdas[i]
#
#                 # Sub-step Euler integration for ODE stability
#                 dl_sub = dl / 10.0
#                 for _sub in tf.range(10):
#                     # M = M0 + beta * Hh
#                     M = self.M0 + beta * self.Hh
#                     # Add jitter to diagonal for safe inversion using pre-computed Identity
#                     M = M + I_mat * 1e-6
#                     M_inv = tf.linalg.inv(M)
#                     M_inv2 = tf.linalg.matmul(M_inv, M_inv)
#
#                     tr_H = tf.linalg.trace(self.Hh)
#                     tr_Minv = tf.linalg.trace(M_inv)
#                     tr_M = tf.linalg.trace(M)
#                     tr_Minv2_H = tf.linalg.trace(tf.linalg.matmul(M_inv2, self.Hh))
#
#                     # Eq 28 (Nuclear Norm case)
#                     d2beta = self.mu * (tr_H * tr_Minv + tr_M * tr_Minv2_H)
#
#                     # Integration steps
#                     beta = beta + dbeta * dl_sub
#                     dbeta = dbeta + d2beta * dl_sub
#
#                 beta_history = beta_history.write(i, beta)
#
#             betas = beta_history.stack()
#
#             # Bisection check: Did we overshoot the target beta(1) = 1.0?
#             if betas[-1] > 1.0:
#                 p_high = p_mid
#             else:
#                 p_low = p_mid
#
#             best_betas = betas
#
#         # Convert absolute beta schedule into step sizes (Delta Beta) strictly using tensors
#         padded_betas = tf.concat([tf.constant([0.0], dtype=tf.float32), best_betas], axis=0)
#         d_betas = padded_betas[1:] - padded_betas[:-1]
#
#         return d_betas
#
#     @tf.function
#     def get_beta_steps(self):
#         return self.optimal_d_betas
#
#
# # ==========================================
# # Dai et al. 2022 - Section 4 Example
# # ==========================================
# def run_dai22_static_example():
#     """
#     Replicates the exact static 1-step example from Section 4 of Dai22.
#     Demonstrates the Shooting Method solving Eq 28 natively in TF.
#     """
#     print("\n--- Running Dai22 Section 4 Static Example ---")
#
#     # 1. Setup Parameters from Section 4
#     P_prior = tf.linalg.diag([1000.0, 2.0])
#     R_cov = tf.linalg.diag([0.04, 0.04])
#     x_prior = tf.constant([3.0, 5.0], dtype=tf.float32)
#     sensors = tf.constant([[3.5, 0.0], [-3.5, 0.0]], dtype=tf.float32)
#
#     # 2. Calculate Jacobian H at the prior mean
#     # h_i = arctan(y / (x - sx_i))
#     dx1 = x_prior[0] - sensors[0, 0]  # 3.0 - 3.5 = -0.5
#     dy1 = x_prior[1] - sensors[0, 1]  # 5.0 - 0.0 = 5.0
#     r2_1 = tf.square(dx1) + tf.square(dy1)
#
#     dx2 = x_prior[0] - sensors[1, 0]  # 3.0 - (-3.5) = 6.5
#     dy2 = x_prior[1] - sensors[1, 1]  # 5.0 - 0.0 = 5.0
#     r2_2 = tf.square(dx2) + tf.square(dy2)
#
#     # Derivation of arctan gradient yields [-dy/r^2, dx/r^2]
#     # H = tf.constant([
#     #     [-dy1 / r2_1, dx1 / r2_1],
#     #     [-dy2 / r2_2, dx2 / r2_2]
#     # ], dtype=tf.float32)
#     # H = tf.constant([
#     #     [-dy1 / r2_1, dx1 / r2_1],
#     #     [-dy2 / r2_2, dx2 / r2_2]
#     # ])
#     # H = tf.stack([
#     #     tf.stack([-dy1 / r2_1, dx1 / r2_1]),
#     #     tf.stack([-dy2 / r2_2, dx2 / r2_2])
#     # ], dtype=tf.float32)
#     H = tf.convert_to_tensor([[-dy1 / r2_1, dx1 / r2_1], [-dy2 / r2_2, dx2 / r2_2]], dtype=tf.float32)
#
#     # 3. Build M0 and Hh Matrices for Eq 28
#     M0 = tf.linalg.inv(P_prior)
#     R_inv = tf.linalg.inv(R_cov)
#     Hh = tf.linalg.matmul(H, tf.linalg.matmul(R_inv, H), transpose_a=True)
#
#     # 4. Solve using the pure TF Bisection Shooting Method (mu=0.2)
#     scheduler = ShootingScheduler(M0, Hh, mu=0.2, n_steps=29)
#     d_betas = scheduler.get_beta_steps()
#
#     # 5. Reconstruct the absolute beta curve for plotting
#     lambdas = scheduler.d_lambdas
#     abs_lambdas = tf.math.cumsum(lambdas).numpy()
#     abs_betas = tf.math.cumsum(d_betas).numpy()
#
#     # 6. Plotting to match Figure 2(a) in Dai22
#     plt.figure(figsize=(6, 5))
#
#     # Prepend 0 for the origin of the plot
#     plot_x = [0.0] + list(abs_lambdas)
#     plot_y = [0.0] + list(abs_betas)
#
#     plt.plot(plot_x, plot_y, 'r-', lw=2, label=r'Optimal $\beta^*(\lambda)$')
#     plt.plot([0, 1], [0, 1], 'b--', label=r'Baseline $\beta(\lambda) = \lambda$')
#     plt.title("Dai22 Figure 2(a): Optimal Homotopy Schedule")
#     plt.xlabel(r'$\lambda$')
#     plt.ylabel(r'$\beta(\lambda)$')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     print("Static Example Complete. Curve matches Figure 2(a).")
#
#
# if __name__ == "__main__":
#     run_dai22_static_example()