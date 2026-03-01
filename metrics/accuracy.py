import tensorflow as tf

@tf.function
def calculate_rmse(estimate, truth):
    """Root Mean Square Error: sqrt(mean((est - truth)^2))"""
    mse = tf.reduce_mean(tf.square(estimate - truth))
    return tf.sqrt(mse)

@tf.function
def calculate_mae(estimate, truth):
    """Mean Absolute Error: mean(|est - truth|)"""
    return tf.reduce_mean(tf.abs(estimate - truth))