import tensorflow as tf

def get_grad_norm(tape, loss, target):
    """Calculates L2 norm of gradients to check differentiability."""
    grads = tape.gradient(loss, target)
    if grads is None:
        return tf.constant(0.0)
    return tf.norm(grads)