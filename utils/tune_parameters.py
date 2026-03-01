import tensorflow as tf
import time
from resamplers.dresamplers import OTResample


def tune_ot_parameters():
    print("=== TUNING OT RESAMPLER (PURE TF) ===")
    print(f"{'Eps':<6} | {'Iters':<5} | {'Cost':<8} | {'Bias(Var_Ratio)':<15} | {'Speed(ms)':<8}")
    print("-" * 55)

    B, N, D = 10, 100, 2
    p1 = tf.random.normal([B, N // 2, D]) - 5.0
    p2 = tf.random.normal([B, N // 2, D]) + 5.0
    particles = tf.concat([p1, p2], axis=1)
    weights = tf.ones([B, N]) / tf.cast(N, tf.float32)

    true_var = tf.math.reduce_variance(particles)

    configs = [(1.0, 5), (0.5, 10), (0.1, 10), (0.05, 10), (0.01, 50)]

    best_ot = (0.1, 10)
    best_ot_score = -99.0

    for eps, iters in configs:
        resampler = OTResample(epsilon=eps, n_iters=iters)

        # Warmup for graph compilation
        _ = resampler(particles, weights)

        start = time.perf_counter()
        # Using unified __call__ interface
        new_parts, _ = resampler(particles, weights)
        elapsed = (time.perf_counter() - start) * 1000

        new_var = tf.math.reduce_variance(new_parts)
        ratio = new_var / (true_var + 1e-10)

        cost = tf.norm(tf.reduce_mean(new_parts, axis=1) - tf.reduce_mean(particles, axis=1))

        # Heuristic: Penalize high cost and explicitly penalize slow execution time
        time_penalty = elapsed * 0.01
        score = ratio - (5.0 * cost) - time_penalty

        marker = "(*)" if score > best_ot_score else ""
        if score > best_ot_score:
            best_ot_score = score
            best_ot = (eps, iters)

        print(f"{eps:<6.2f} | {iters:<5} | {cost:<8.4f} | {ratio:<15.4f} | {elapsed:<8.2f} {marker}")

    print(f"\nSelected Tuned Params -> OT: {best_ot}")
    return best_ot