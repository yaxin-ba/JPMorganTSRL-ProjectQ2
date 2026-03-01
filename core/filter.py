import tensorflow as tf


class BaseFilter(tf.Module):
    def __init__(self, ssm, name=None):
        super().__init__(name=name)
        self.ssm = ssm

    @tf.function
    def predict(self, state, params, t):
        raise NotImplementedError

    @tf.function
    def update(self, state, obs, params, t):
        raise NotImplementedError


if __name__ == "__main__":
    print("Testing core/filter.py...")


    class DummyFilter(BaseFilter):
        @tf.function
        def predict(self, state, params, t):
            return state + 1.0

        @tf.function
        def update(self, state, obs, params, t):
            return state + obs


    dummy_filter = DummyFilter(ssm=None)
    s = tf.constant(0.0)

    s_pred = dummy_filter.predict(s, None, 0)
    s_upd = dummy_filter.update(s_pred, tf.constant(5.0), None, 0)

    print("Predicted state:", s_pred.numpy())
    print("Updated state:", s_upd.numpy())
    print("core/filter.py passed.\n")