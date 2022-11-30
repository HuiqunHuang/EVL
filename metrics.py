from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mul


@tf.function
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

@tf.function
def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

@tf.function
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# aliases
mse = MSE = mean_squared_error
# rmse = RMSE = root_mean_square_error

@tf.function
def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))

@tf.function
def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5
@tf.function
def EVL(y_true, y_pred):
    y = 3
    beta1 = tf.reduce_sum(y_true, axis=-1)
    beta1 = beta1 / len(y_true[0])
    beta0 = 1 - beta1
    beta0 = tf.cast(beta0, tf.float32)
    beta1 = tf.cast(beta1, tf.float32)
    loss = K.mean((-tf.multiply(K.pow(1 - y_pred / y, y), tf.expand_dims(beta0, 1)) * y_true * K.log(y_pred + 0.01) - tf.multiply(K.pow(
        1 - (1 - y_pred) / y, y), tf.expand_dims(beta1, 1)) * (1 - y_true) * K.log(1 - y_pred + 0.01)))

    return loss

@tf.function
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

mae = MAE = mean_absolute_error