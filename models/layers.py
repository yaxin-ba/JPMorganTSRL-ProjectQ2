import tensorflow as tf
import math


# class Linear(tf.Module):
#     def __init__(self, in_features, out_features, activation=None):
#         super().__init__()
#         stddev = 1.0 / math.sqrt(in_features)
#         w_init = tf.random.normal([in_features, out_features], stddev=stddev)
#         self.w = tf.Variable(w_init, name='w')
#         self.b = tf.Variable(tf.zeros([out_features]), name='b')
#         self.activation = activation
#
#     @tf.function
#     def __call__(self, x):
#         out = tf.matmul(x, self.w) + self.b
#         if self.activation:
#             out = self.activation(out)
#         return out
#
#
# class ConditionalAffineFlow(tf.Module):
#     def __init__(self, input_dim=1, cond_dim=2):
#         super().__init__()
#         self.l1 = Linear(input_dim + cond_dim, 32, activation=tf.nn.tanh)
#         self.l2 = Linear(32, 32, activation=tf.nn.tanh)
#         self.l3 = Linear(32, 2, activation=None)
#
#         self.l3.w.assign(self.l3.w * 0.01)
#         self.l3.b.assign(self.l3.b * 0.01)
#
#     @tf.function
#     def __call__(self, z, condition):
#         inp = tf.concat([z, condition], axis=2)
#         h = self.l1(inp)
#         h = self.l2(h)
#         params = self.l3(h)
#
#         log_scale, shift = tf.split(params, num_or_size_splits=2, axis=2)
#         log_scale = tf.clip_by_value(log_scale, -5.0, 2.0)
#
#         return z * tf.exp(log_scale) + shift, log_scale
#
#
# class PositiveLinear(tf.Module):
#     """Linear layer strictly enforcing positive weights for convexity."""
#     def __init__(self, in_dim, out_dim, use_bias=True, activation=None, name=None):
#         super().__init__(name=name)
#         self.use_bias = use_bias
#         init = tf.random.normal([in_dim, out_dim], stddev=0.1)
#         self.w_raw = tf.Variable(init, name="w_raw")
#         if use_bias: self.b = tf.Variable(tf.zeros([out_dim]), name="b")
#         self.activation = activation
#
#     @tf.function
#     def __call__(self, x):
#         y = tf.matmul(x, tf.math.softplus(self.w_raw))
#         if self.use_bias: y += self.b
#         return self.activation(y) if self.activation else y
#
#
# class ConvexDeepONet(tf.Module):
#     """DeepONet architecture preserving Input Convexity (ICNN)."""
#     def __init__(self, x_dim=1, context_dim=2, basis_dim=32):
#         super().__init__()
#         self.branch_l1 = Linear(context_dim, 32, activation=tf.nn.tanh)
#         self.branch_l2 = Linear(32, 32, activation=tf.nn.tanh)
#         self.branch_l3 = Linear(32, basis_dim, activation=tf.math.softplus)
#
#         self.trunk_u_skip = Linear(x_dim, 32, use_bias=True)
#         self.trunk_u_out = Linear(x_dim, basis_dim, use_bias=False)
#         self.trunk_z_layer = PositiveLinear(32, 32, use_bias=True)
#         self.trunk_u_layer = Linear(x_dim, 32, use_bias=False)
#         self.trunk_z_out = PositiveLinear(32, basis_dim, use_bias=True)
#
#     @tf.function
#     def trunk_forward(self, x):
#         z = tf.nn.softplus(self.trunk_u_skip(x))
#         z = tf.nn.softplus(self.trunk_z_layer(z) + self.trunk_u_layer(x))
#         return self.trunk_z_out(z) + self.trunk_u_out(x)
#
#     @tf.function
#     def __call__(self, x, context):
#         w = self.branch_l3(self.branch_l2(self.branch_l1(context)))
#         phi = self.trunk_forward(x)
#         return tf.reduce_sum(w * phi, axis=-1, keepdims=True)
#
#
# class ICNN(tf.Module):
#     """Standard Input Convex Neural Network."""
#     def __init__(self, x_dim, ctx_dim, hidden_dim=32, name="icnn"):
#         super().__init__(name=name)
#         input_dim = x_dim + ctx_dim
#         self.w0 = Linear(input_dim, hidden_dim, name="w0")
#         self.z1_z = PositiveLinear(hidden_dim, hidden_dim, name="z1_z")
#         self.z1_u = Linear(input_dim, hidden_dim, name="z1_u")
#         self.z2_z = PositiveLinear(hidden_dim, hidden_dim, name="z2_z")
#         self.z2_u = Linear(input_dim, hidden_dim, name="z2_u")
#         self.out_z = PositiveLinear(hidden_dim, 1, name="out_z")
#         self.out_u = Linear(input_dim, 1, name="out_u")
#
#     @tf.function
#     def __call__(self, x, ctx):
#         u = tf.concat([x, ctx], axis=2)
#         z = tf.nn.softplus(self.w0(u))
#         z = tf.nn.softplus(self.z1_z(z) + self.z1_u(u))
#         z = tf.nn.softplus(self.z2_z(z) + self.z2_u(u))
#         return self.out_z(z) + self.out_u(u)
#
#
# class OT_TransportMap(tf.Module):
#     """Wrapper that turns a convex potential (ICNN/DeepONet) into a Transport gradient map."""
#     def __init__(self, core_network):
#         super().__init__()
#         self.net = core_network
#
#     @tf.function
#     def __call__(self, x, context):
#         with tf.GradientTape() as tape:
#             tape.watch(x)
#             psi = self.net(x, context)
#         grad = tape.gradient(psi, x)
#         if grad is None: grad = tf.zeros_like(x)
#         return x + 0.1 * grad, psi

# ==========================================
# 1. Base Layers
# ==========================================

class Linear(tf.Module):
    def __init__(self, in_features, out_features, use_bias=True, activation=None, name=None):
        super().__init__(name=name)
        self.use_bias = use_bias
        stddev = 1.0 / math.sqrt(in_features)
        w_init = tf.random.normal([in_features, out_features], stddev=stddev)
        self.w = tf.Variable(w_init, name='w')
        if self.use_bias:
            self.b = tf.Variable(tf.zeros([out_features]), name='b')
        self.activation = activation

    @tf.function
    def __call__(self, x):
        out = tf.matmul(x, self.w)
        if self.use_bias:
            out += self.b
        if self.activation:
            out = self.activation(out)
        return out


class PositiveLinear(tf.Module):
    """Linear layer strictly enforcing positive weights for convexity."""

    def __init__(self, in_dim, out_dim, use_bias=True, activation=None, name=None):
        super().__init__(name=name)
        self.use_bias = use_bias
        init = tf.random.normal([in_dim, out_dim], stddev=0.1)
        self.w_raw = tf.Variable(init, name="w_raw")
        if use_bias:
            self.b = tf.Variable(tf.zeros([out_dim]), name="b")
        self.activation = activation

    @tf.function
    def __call__(self, x):
        y = tf.matmul(x, tf.math.softplus(self.w_raw))
        if self.use_bias:
            y += self.b
        return self.activation(y) if self.activation else y


# ==========================================
# 2. Advanced Network Architectures
# ==========================================

class ConditionalAffineFlow(tf.Module):
    def __init__(self, input_dim=1, cond_dim=2):
        super().__init__()
        self.l1 = Linear(input_dim + cond_dim, 32, activation=tf.nn.tanh)
        self.l2 = Linear(32, 32, activation=tf.nn.tanh)
        self.l3 = Linear(32, 2, activation=None)

        self.l3.w.assign(self.l3.w * 0.01)
        if self.l3.use_bias:
            self.l3.b.assign(self.l3.b * 0.01)

    @tf.function
    def __call__(self, z, condition):
        inp = tf.concat([z, condition], axis=2)
        h = self.l1(inp)
        h = self.l2(h)
        params = self.l3(h)

        log_scale, shift = tf.split(params, num_or_size_splits=2, axis=2)
        log_scale = tf.clip_by_value(log_scale, -5.0, 2.0)

        return z * tf.exp(log_scale) + shift, log_scale


# class ConvexDeepONet(tf.Module):
#     """DeepONet architecture preserving Input Convexity (ICNN)."""
#
#     def __init__(self, x_dim=1, context_dim=2, basis_dim=32):
#         super().__init__()
#         self.branch_l1 = Linear(context_dim, 32, activation=tf.nn.tanh)
#         self.branch_l2 = Linear(32, 32, activation=tf.nn.tanh)
#         # self.branch_l3 = Linear(32, basis_dim, activation=tf.math.softplus)
#         self.branch_l3 = Linear(32, basis_dim, activation=lambda x: tf.math.softplus(x) + 1e-3)
#
#         self.trunk_u_skip = Linear(x_dim, 32, use_bias=True)
#         self.trunk_u_out = Linear(x_dim, basis_dim, use_bias=False)
#         self.trunk_z_layer = PositiveLinear(32, 32, use_bias=True)
#         self.trunk_u_layer = Linear(x_dim, 32, use_bias=False)
#         self.trunk_z_out = PositiveLinear(32, basis_dim, use_bias=True)
#
#     @tf.function
#     def trunk_forward(self, x):
#         z = tf.nn.softplus(self.trunk_u_skip(x))
#         z = tf.nn.softplus(self.trunk_z_layer(z) + self.trunk_u_layer(x))
#         return self.trunk_z_out(z) + self.trunk_u_out(x)
#
#     @tf.function
#     def __call__(self, x, context):
#         w = self.branch_l3(self.branch_l2(self.branch_l1(context)))
#         phi = self.trunk_forward(x)
#         return tf.reduce_sum(w * phi, axis=-1, keepdims=True)


class ICNN(tf.Module):
    """Standard Input Convex Neural Network."""

    def __init__(self, x_dim, ctx_dim, hidden_dim=32, name="icnn"):
        super().__init__(name=name)
        input_dim = x_dim + ctx_dim
        self.w0 = Linear(input_dim, hidden_dim, name="w0")
        self.z1_z = PositiveLinear(hidden_dim, hidden_dim, name="z1_z")
        self.z1_u = Linear(input_dim, hidden_dim, name="z1_u")
        self.z2_z = PositiveLinear(hidden_dim, hidden_dim, name="z2_z")
        self.z2_u = Linear(input_dim, hidden_dim, name="z2_u")
        self.out_z = PositiveLinear(hidden_dim, 1, name="out_z")
        self.out_u = Linear(input_dim, 1, name="out_u")

    @tf.function
    def __call__(self, x, ctx):
        u = tf.concat([x, ctx], axis=2)
        z = tf.nn.softplus(self.w0(u))
        z = tf.nn.softplus(self.z1_z(z) + self.z1_u(u))
        z = tf.nn.softplus(self.z2_z(z) + self.z2_u(u))
        return self.out_z(z) + self.out_u(u)


# ==========================================
# 3. Transport Map Wrapper
# ==========================================

class ConvexDeepONet(tf.Module):
    """DeepONet architecture preserving Input Convexity (ICNN)."""

    def __init__(self, x_dim=1, context_dim=2, basis_dim=32):
        super().__init__()
        self.branch_l1 = Linear(context_dim, 32, activation=tf.nn.tanh)
        self.branch_l2 = Linear(32, 32, activation=tf.nn.tanh)

        # Reverted to your standard layer signature. No initializer tricks.
        self.branch_l3 = Linear(32, basis_dim, activation=tf.math.softplus)

        self.trunk_u_skip = Linear(x_dim, 32, use_bias=True)
        self.trunk_u_out = Linear(x_dim, basis_dim, use_bias=False)
        self.trunk_z_layer = PositiveLinear(32, 32, use_bias=True)
        self.trunk_u_layer = Linear(x_dim, 32, use_bias=False)
        self.trunk_z_out = PositiveLinear(32, basis_dim, use_bias=True)

    def trunk_forward(self, x):
        z = tf.nn.softplus(self.trunk_u_skip(x))
        z = tf.nn.softplus(self.trunk_z_layer(z) + self.trunk_u_layer(x))
        return self.trunk_z_out(z) + self.trunk_u_out(x)

    def __call__(self, x, context):
        w = self.branch_l3(self.branch_l2(self.branch_l1(context)))
        phi = self.trunk_forward(x)
        return tf.reduce_sum(w * phi, axis=-1, keepdims=True)


class OT_TransportMap(tf.Module):
    """Strict Brenier Transport Map: T(x) = x + grad(psi)."""

    def __init__(self, core_network):
        super().__init__()
        self.net = core_network

    def __call__(self, x, context):
        with tf.GradientTape() as tape:
            tape.watch(x)
            psi = self.net(x, context)

        grad = tape.gradient(psi, x)
        if grad is None:
            grad = tf.zeros_like(x)

        # Pure theoretical Monge-Ampère map
        return x + grad, psi