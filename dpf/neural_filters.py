import tensorflow as tf
from models.layers import Linear, ConditionalAffineFlow


class SIS_Filter(tf.Module):
    """Sequential Importance Sampling (No Resampling)"""

    def __init__(self, ssm, n_particles=50, name="SIS"):
        super().__init__(name=name)
        self.ssm = ssm
        self.N = tf.constant(n_particles, dtype=tf.int32)

    @tf.function
    def __call__(self, observations):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]
        D = tf.shape(observations)[2]
        N_float = tf.cast(self.N, tf.float32)

        particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
        log_weights = tf.zeros([B, self.N]) - tf.math.log(N_float)

        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            obs = observations[:, t]
            t_f = tf.cast(t + 1, tf.float32)

            noise = tf.random.normal(tf.shape(particles)) * tf.sqrt(self.ssm.Q[0, 0])
            particles = self.ssm.f_fn(particles, t_f) + noise

            pred_obs = self.ssm.h_fn(particles, t_f)
            dist = tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs, 1)), axis=2)
            log_lik = -0.5 * dist / self.ssm.R[0, 0]

            log_weights = log_weights + log_lik
            lse = tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
            norm_weights = tf.exp(log_weights - lse)

            est = tf.reduce_sum(tf.expand_dims(norm_weights, 2) * particles, axis=1)
            est_states = est_states.write(t, est)

        return tf.transpose(est_states.stack(), perm=[1, 0, 2]), tf.ones([B, T])


class NeuralProposal_Filter(tf.Module):
    """DPF with Learned Neural Proposal Network"""

    def __init__(self, ssm, resampler, n_particles=50, name="NeuralPF"):
        super().__init__(name=name)
        self.ssm = ssm
        self.resampler = resampler
        self.N = tf.constant(n_particles, dtype=tf.int32)

        self.l1 = Linear(2, 32, activation=tf.nn.tanh)
        self.l2 = Linear(32, 1, activation=None)

        self.l2.w.assign(tf.zeros_like(self.l2.w))
        self.l2.b.assign(tf.zeros_like(self.l2.b))

    @tf.function
    def __call__(self, observations):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]
        D = tf.shape(observations)[2]

        particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
        weights = tf.ones([B, self.N]) / tf.cast(self.N, tf.float32)

        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            obs = observations[:, t]
            t_f = tf.cast(t + 1, tf.float32)

            if t > 0 and self.resampler is not None:
                particles, weights = self.resampler(particles, weights)

            obs_exp = tf.tile(tf.expand_dims(obs, 1), [1, self.N, 1])
            inp = tf.concat([particles, obs_exp], axis=2)

            shift = tf.clip_by_value(self.l2(self.l1(inp)), -3.0, 3.0)

            trans_mean = self.ssm.f_fn(particles, t_f)
            std = tf.sqrt(self.ssm.Q[0, 0])

            noise = tf.random.normal(tf.shape(particles)) * std
            particles_prop = particles + shift + noise

            pred_obs = self.ssm.h_fn(particles_prop, t_f)
            dist = tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs, 1)), axis=2)
            log_lik = -0.5 * dist / self.ssm.R[0, 0]

            log_p = -0.5 * tf.reduce_sum(tf.square(particles_prop - trans_mean), axis=2) / (std ** 2)
            log_q = -0.5 * tf.reduce_sum(tf.square(particles_prop - (particles + shift)), axis=2) / (std ** 2)

            log_w = log_lik + log_p - log_q
            log_w = tf.clip_by_value(log_w, -20.0, 20.0)

            weights = weights * tf.exp(log_w) + 1e-10
            weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights), weights)
            weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True))

            est = tf.reduce_sum(tf.expand_dims(weights, 2) * particles_prop, axis=1)
            est_states = est_states.write(t, est)
            particles = particles_prop

        return tf.transpose(est_states.stack(), perm=[1, 0, 2]), tf.ones([B, T])


class NF_DPF_Filter(tf.Module):
    """DPF with Normalizing Flow Proposal"""

    def __init__(self, ssm, resampler, n_particles=50, name="NF_DPF"):
        super().__init__(name=name)
        self.ssm = ssm
        self.resampler = resampler
        self.N = tf.constant(n_particles, dtype=tf.int32)
        self.flow = ConditionalAffineFlow(1, 2)

    @tf.function
    def __call__(self, observations):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]
        D = tf.shape(observations)[2]

        particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
        weights = tf.ones([B, self.N]) / tf.cast(self.N, tf.float32)
        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            obs = observations[:, t]
            t_f = tf.cast(t + 1, tf.float32)

            if t > 0 and self.resampler is not None:
                particles, weights = self.resampler(particles, weights)

            z = tf.random.normal(tf.shape(particles))
            obs_exp = tf.tile(tf.expand_dims(obs, 1), [1, self.N, 1])
            cond = tf.concat([particles, obs_exp], axis=2)

            particles_new, log_det = self.flow(z, cond)

            pred_obs = self.ssm.h_fn(particles_new, t_f)
            dist = tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs, 1)), axis=2)
            log_lik = -0.5 * dist / self.ssm.R[0, 0]

            std = tf.sqrt(self.ssm.Q[0, 0])
            trans_mean = self.ssm.f_fn(particles, t_f)

            log_p = -0.5 * tf.reduce_sum(tf.square(particles_new - trans_mean), axis=2) / (std ** 2)
            log_p_z = -0.5 * tf.reduce_sum(tf.square(z), axis=2) - 0.9189

            log_q = log_p_z - tf.squeeze(log_det, axis=2)

            log_w = log_lik + log_p - log_q
            log_w = tf.clip_by_value(log_w, -20.0, 20.0)

            weights = weights * tf.exp(log_w) + 1e-10
            weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights), weights)
            weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True))

            est = tf.reduce_sum(tf.expand_dims(weights, 2) * particles_new, axis=1)
            est_states = est_states.write(t, est)
            particles = particles_new

        return tf.transpose(est_states.stack(), perm=[1, 0, 2]), tf.ones([B, T])


class Transport_DPF_Filter(tf.Module):
    """
    Evaluates Neural OT networks by transitioning particles to the prior,
    transporting them to the posterior, and then reweighting/resampling.
    """
    def __init__(self, ssm, resampler, transport_net, n_particles=100, name="TransportPF"):
        super().__init__(name=name)
        self.ssm = ssm
        self.resampler = resampler
        self.N = tf.constant(n_particles, dtype=tf.int32)
        self.transport_net = transport_net

    @tf.function(reduce_retracing=True)
    def __call__(self, observations):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]
        D = tf.shape(observations)[2]
        T_float = tf.cast(T, tf.float32)

        particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
        weights = tf.ones([B, self.N]) / tf.cast(self.N, tf.float32)
        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            obs = observations[:, t]
            t_f = tf.cast(t + 1, tf.float32)

            # 1. Transition (Generate Prior)
            noise = tf.random.normal(tf.shape(particles)) * tf.sqrt(self.ssm.Q[0, 0])
            particles_prior = self.ssm.f_fn(particles, t_f) + noise

            # 2. Transport (Map to Posterior via Neural OT)
            obs_exp = tf.tile(tf.expand_dims(obs, 1), [1, self.N, 1])
            ctx = tf.concat([obs_exp, tf.ones_like(obs_exp) * (t_f / T_float)], axis=2)
            particles_post, _ = self.transport_net(particles_prior, ctx)

            # 3. Weight Evaluation
            pred_obs = self.ssm.h_fn(particles_post, t_f)
            log_lik = -0.5 * tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs, 1)), axis=2) / self.ssm.R[0, 0]
            log_w = tf.math.log(weights + 1e-16) + log_lik
            weights = tf.exp(log_w - tf.reduce_logsumexp(log_w, axis=1, keepdims=True))
            weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-10)

            # 4. State Estimate
            est = tf.reduce_sum(tf.expand_dims(weights, 2) * particles_post, axis=1)
            est_states = est_states.write(t, est)

            # 5. Differentiable Resampling (Cures degeneracy while keeping gradients)
            if self.resampler is not None:
                particles, weights = self.resampler(particles_post, weights)
            else:
                particles = particles_post

        return tf.transpose(est_states.stack(), perm=[1, 0, 2]), tf.ones([B, T])


class NeuralOT_Resampler_Filter(tf.Module):
    """
    Filter where the neural network completely replaces the OT resampler.
    It deterministically maps the unweighted prior directly to the posterior.
    """

    def __init__(self, ssm, transport_net, n_particles=100, name="NeuralOT_Resampler"):
        super().__init__(name=name)
        self.ssm = ssm
        self.N = tf.constant(n_particles, dtype=tf.int32)
        self.transport_net = transport_net

    @tf.function(reduce_retracing=True)
    def __call__(self, observations):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]
        D = tf.shape(observations)[2]
        T_float = tf.cast(T, tf.float32)

        # Initialize particles
        particles = tf.random.normal([B, self.N, D]) * tf.sqrt(5.0)
        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            obs = observations[:, t]
            t_f = tf.cast(t + 1, tf.float32)

            # 1. Transition (Generate Prior Distribution)
            noise = tf.random.normal(tf.shape(particles)) * tf.sqrt(self.ssm.Q[0, 0])
            particles_prior = self.ssm.f_fn(particles, t_f) + noise

            # 2. Neural OT Resampling (Map directly to Posterior Distribution)
            obs_exp = tf.tile(tf.expand_dims(obs, 1), [1, self.N, 1])
            ctx = tf.concat([obs_exp, tf.ones_like(obs_exp) * (t_f / T_float)], axis=2)

            particles_post, _ = self.transport_net(particles_prior, ctx)

            # 3. State Estimation (Weights are implicitly uniform 1/N)
            est = tf.reduce_mean(particles_post, axis=1)
            est_states = est_states.write(t, est)

            # 4. Step forward
            particles = particles_post

        # Return estimates and dummy uniform weights for metric compatibility
        return tf.transpose(est_states.stack(), perm=[1, 0, 2]), tf.ones([B, T])