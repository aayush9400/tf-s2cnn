import tensorflow as tf

def as_complex(x):
    """
    In TensorFlow, a complex array is represented as a real array with an extra axis of length 2.
    This function takes a real-valued array x and adds a complex axis where the real part is set to x and the imaginary part is set to 0.
    """
    imaginary = tf.zeros_like(x)
    z = tf.stack((x, imaginary), axis=x.shape.ndims)
    return z

def complex_mm(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [i, k, complex] (M, K, 2)
    :param y: [k, j, complex] (K, N, 2)
    :return:  [i, j, complex] (M, N, 2)
    '''
    x = tf.cast(x, y.dtype)
    xr = x[:, :, 0]
    xi = x[:, :, 1]

    yr = y[:, :, 0]
    yi = y[:, :, 1]

    if not conj_x and not conj_y:
        zr = tf.matmul(xr, yr) - tf.matmul(xi, yi)
        zi = tf.matmul(xr, yi) + tf.matmul(xi, yr)
    if conj_x and not conj_y:
        zr = tf.matmul(xr, yr) + tf.matmul(xi, yi)
        zi = tf.matmul(xr, yi) - tf.matmul(xi, yr)
    if not conj_x and conj_y:
        zr = tf.matmul(xr, yr) + tf.matmul(xi, yi)
        zi = tf.matmul(xi, yr) - tf.matmul(xr, yi)
    if conj_x and conj_y:
        zr = tf.matmul(xr, yr) - tf.matmul(xi, yi)
        zi = -tf.matmul(xr, yi) - tf.matmul(xi, yr)

    return tf.stack((zr, zi), axis=2)
