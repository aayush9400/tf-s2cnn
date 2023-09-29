# pylint: disable=R,C,E1101
import tensorflow as tf
from functools import lru_cache
from s2cnn.utils.decorator import show_running


def so3_integrate(x):
    """
    Integrate a signal on SO(3) using the Haar measure
    
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    :return y: [...] (...)
    """
    # print('so3_integrate', x.shape)
    assert x.shape[-1] == x.shape[-2]
    assert x.shape[-2] == x.shape[-3]

    b = x.shape[-1] // 2

    w = _setup_so3_integrate(b)  # [beta]

    x = tf.reduce_sum(x, axis=-1)  # [..., beta, alpha]
    x = tf.reduce_sum(x, axis=-1)  # [..., beta]

    sz = x.shape.as_list()
    x = tf.reshape(x, [-1, 2 * b])
    w = tf.reshape(w, [2 * b, 1])
    x = tf.squeeze(tf.matmul(x, w), axis=-1)
    x = tf.reshape(x, sz[:-1])
    return x


@lru_cache(maxsize=32)
@show_running
def _setup_so3_integrate(b):
    import lie_learn.spaces.S3 as S3

    weights = S3.quadrature_weights(b)

    return tf.convert_to_tensor(weights, dtype=tf.float32)  # (2b) [beta]  # pylint: disable=E1102
