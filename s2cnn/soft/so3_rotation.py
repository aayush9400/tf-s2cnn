# pylint: disable=C,R,E1101
import tensorflow as tf
import numpy as np

from .so3_fft import SO3_fft_real, SO3_ifft_real
from s2cnn.utils.complex import complex_mm
from functools import lru_cache
from s2cnn.utils.decorator import cached_dirpklgz


def so3_rotation(x, alpha, beta, gamma):
    '''
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    '''
    b = x.shape[-1] // 2
    x_size = x.shape.as_list()

    Us = _setup_so3_rotation(b, alpha, beta, gamma)

    # fourier transform
    x = SO3_fft_real(x, b_out=None)  # [l * m * n, ..., complex]

    # rotated spectrum
    Fz_list = []
    begin = 0
    for l in range(b):
        L = 2 * l + 1
        size = L ** 2

        Fx = x[begin:begin+size]
        Fx = tf.reshape(Fx, (L, -1, 2))  # [m, n * batch, complex]

        U = tf.reshape(Us[l], (L, L, 2))  # [m, n, complex]

        Fz = complex_mm(U, Fx, conj_x=True)  # [m, n * batch, complex]

        Fz = tf.reshape(Fz, (size, -1, 2))  # [m * n, batch, complex]
        Fz_list.append(Fz)

        begin += size

    Fz = tf.concat(Fz_list, axis=0)  # [l * m * n, batch, complex]
    z = SO3_ifft_real(Fz, b_out=None)

    z = tf.reshape(z, x_size)

    return z


@cached_dirpklgz("cache/setup_so3_rotation")
def __setup_so3_rotation(b, alpha, beta, gamma):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    Us = [wigner_D_matrix(l, alpha, beta, gamma,
                          field='complex', normalization='quantum', order='centered', condon_shortley='cs')
          for l in range(b)]
    # Us[l][m, n] = exp(i m alpha) d^l_mn(beta) exp(i n gamma)

    Us = [Us[l].astype(np.complex64).view(np.float32).reshape((2 * l + 1, 2 * l + 1, 2)) for l in range(b)]

    return Us


@lru_cache(maxsize=32)
def _setup_so3_rotation(b, alpha, beta, gamma):
    Us = __setup_so3_rotation(b, alpha, beta, gamma)

    # convert to tensorflow Tensor
    Us = [tf.convert_to_tensor(U, dtype=tf.float32) for U in Us]  # pylint: disable=E1102

    return Us
