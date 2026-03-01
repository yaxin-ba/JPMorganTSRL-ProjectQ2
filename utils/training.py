import tensorflow as tf




def train_model(model, obs_batch, true_states, epochs=50):
    if len(model.trainable_variables) == 0:
        return model

    optimizer = tf.optimizers.Adam(learning_rate=0.005)

    @tf.function
    def train_step(obs, states):
        with tf.GradientTape() as tape:
            est, _ = model(obs)
            loss = tf.reduce_mean(tf.square(est - states))

        if not tf.math.is_nan(loss):
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    for _ in range(epochs):
        loss = train_step(obs_batch, true_states)

    return model


def train_gradnet_ot(model, obs_seq, states_seq, epochs=150):
    """Strict Algorithm 1 (Chaudhari25). Pure Monge-Ampère equation."""
    if len(model.trainable_variables) == 0: return model
    opt = tf.optimizers.Adam(learning_rate=0.005)
    T_max, T_float = tf.shape(obs_seq)[1], tf.cast(tf.shape(obs_seq)[1], tf.float32)

    @tf.function
    def train_step():
        t_start = tf.random.uniform([], 0, T_max - 6, dtype=tf.int32)
        true_start = states_seq[:, t_start]
        x_source = true_start + tf.random.normal([tf.shape(obs_seq)[0], model.N, 1]) * 2.0
        total_ma_loss = 0.0

        for k in tf.range(5):
            t, t_f = t_start + k, tf.cast(t_start + k + 1, tf.float32)
            obs_t = obs_seq[:, t]
            ctx = tf.concat([tf.tile(tf.expand_dims(obs_t, 1), [1, model.N, 1]),
                             tf.ones([tf.shape(obs_seq)[0], model.N, 1]) * (t_f / T_float)], axis=2)

            with tf.GradientTape() as tape_j:
                tape_j.watch(x_source)
                y_target, _ = model.transport_net(x_source, ctx)

            log_det_J = tf.math.log(tf.maximum(tape_j.gradient(y_target, x_source), 1e-6))

            log_p_x = -0.5 * tf.reduce_sum(tf.square(x_source - true_start), axis=2) / 4.0
            log_lik = -0.5 * tf.reduce_sum(tf.square(model.ssm.h_fn(y_target, t_f) - tf.expand_dims(obs_t, 1)),
                                           axis=2) / model.ssm.R[0, 0]
            log_prior = -0.5 * tf.reduce_sum(tf.square(y_target - model.ssm.f_fn(x_source, t_f)), axis=2) / model.ssm.Q[
                0, 0]

            ma_residual = log_det_J - (tf.expand_dims(log_p_x, 2) - tf.expand_dims(log_lik + log_prior, 2))
            total_ma_loss += tf.reduce_mean(tf.square(ma_residual))

            x_source, true_start = tf.stop_gradient(model.ssm.f_fn(y_target, t_f)), states_seq[:, t]

        return total_ma_loss

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            loss = train_step()
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_variables), 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    return model


def train_deeponet_ot(model, obs_seq, states_seq, epochs=150):
    """DeepONet strictly trained via PINO Loss with no artificial boundaries."""
    if len(model.trainable_variables) == 0: return model
    opt = tf.optimizers.Adam(learning_rate=0.005)
    T_max, T_float = tf.shape(obs_seq)[1], tf.cast(tf.shape(obs_seq)[1], tf.float32)
    lambda_1, lambda_2 = 0.1, 0.05

    @tf.function
    def train_step():
        t_start = tf.random.uniform([], 0, T_max - 6, dtype=tf.int32)
        p = states_seq[:, t_start] + tf.random.normal([tf.shape(obs_seq)[0], model.N, 1]) * 2.0
        total_loss = 0.0

        for k in tf.range(5):
            t, t_f = t_start + k, tf.cast(t_start + k + 1, tf.float32)
            obs_t = obs_seq[:, t]
            ctx = tf.concat([tf.tile(tf.expand_dims(obs_t, 1), [1, model.N, 1]),
                             tf.ones([tf.shape(obs_seq)[0], model.N, 1]) * (t_f / T_float)], axis=2)

            with tf.GradientTape() as tape_hess:
                tape_hess.watch(p)
                p_new, _ = model.transport_net(p, ctx)

            grad_T = tf.maximum(tape_hess.gradient(p_new, p), 1e-6)
            loss_entropy = -lambda_1 * tf.reduce_mean(tf.math.log(grad_T))

            pred_obs = model.ssm.h_fn(p_new, t_f)
            loss_lik = tf.reduce_mean(
                0.5 * tf.reduce_sum(tf.square(pred_obs - tf.expand_dims(obs_t, 1)), axis=2) / model.ssm.R[0, 0])
            loss_reg = lambda_2 * tf.reduce_mean(tf.square(p_new - p))

            total_loss += loss_lik + loss_entropy + loss_reg
            p = tf.stop_gradient(model.ssm.f_fn(p_new, t_f))

        return total_loss

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            loss = train_step()
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_variables), 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    return model
