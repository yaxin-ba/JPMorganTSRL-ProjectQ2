import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class StateSpaceModel(tf.Module):
    def __init__(self, state_dim, obs_dim, name=None):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    def transition(self, x_prev, t):
        """Returns tfd.Distribution for x_t | x_{t-1}"""
        raise NotImplementedError

    def observation(self, x_t, t):
        """Returns tfd.Distribution for y_t | x_t"""
        raise NotImplementedError


if __name__ == "__main__":
    print("Testing core/ssm.py...")


    # Dummy implementation to test availability
    class DummySSM(StateSpaceModel):
        def transition(self, x_prev, t):
            return tfd.Normal(loc=x_prev, scale=1.0)

        def observation(self, x_t, t):
            return tfd.Normal(loc=x_t, scale=0.5)


    ssm = DummySSM(state_dim=2, obs_dim=2)
    dummy_state = tf.constant([1.0, 2.0])

    # Test transition and observation
    next_state_dist = ssm.transition(dummy_state, 0)
    obs_dist = ssm.observation(dummy_state, 0)

    print("Transition Sample:", next_state_dist.sample())
    print("Observation Sample:", obs_dist.sample())
    print("core/ssm.py passed.\n")