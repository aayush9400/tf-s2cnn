# pylint: disable=C,R,E1101
import math
import tensorflow as tf

from .so3_fft import SO3_fft_real, SO3_ifft_real
from s2cnn import so3_mm
from s2cnn import so3_rft


class SO3Convolution(tf.keras.layers.Layer):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the SO(3) group defining the kernel, tuple of (alpha, beta, gamma)'s
        '''
        super(SO3Convolution, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid

        # When useing ADAM optimizer, the variance of each componant of the gradient
        # is normalized by ADAM around 1.
        # Then it is suited to have parameters of order one.
        # Therefore the scaling, needed for the proper forward propagation, is done "outside" of the parameters
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))

    def build(self, input_shape):
        k_init = tf.random_uniform_initializer(minval=-1,maxval=1)
        # self.kernel = tf.Variable(initial_value=k_init(shape=[self.nfeature_in, self.nfeature_out, len(self.grid)], dtype=tf.float32), trainable=True)

        b_init = tf.zeros_initializer()
        # self.bias = tf.Variable(initial_value=b_init(shape=[1, self.nfeature_out, 1, 1, 1], dtype=tf.float32), trainable=True)

        self.kernel = self.add_weight(shape=[self.nfeature_in, self.nfeature_out, len(self.grid)], initializer=k_init, trainable=True)
        self.bias = self.add_weight(shape=[1, self.nfeature_out, 1, 1, 1], initializer=b_init, trainable=True)


    def call(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.shape[1] == self.nfeature_in
        assert x.shape[2] == 2 * self.b_in
        assert x.shape[3] == 2 * self.b_in
        assert x.shape[4] == 2 * self.b_in

        x = SO3_fft_real(x, self.b_out)  # [l * m * n, batch, feature_in, complex]
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out, complex]
        assert x.shape[0] == y.shape[0]
        assert x.shape[2] == y.shape[1]
        z = so3_mm(x, y)  # [l * m * n, batch, feature_out, complex]
        assert z.shape[0] == x.shape[0]
        assert z.shape[1] == x.shape[1]
        assert z.shape[2] == y.shape[2]
        z = SO3_ifft_real(z, b_out=None)  # [batch, feature_out, beta, alpha, gamma]

        z = z + self.bias
        # print("so3 layer", z.shape.as_list())
        return z


class SO3Shortcut(tf.keras.layers.Layer):
    '''
    Useful for ResNet
    '''

    def __init__(self, nfeature_in, nfeature_out, b_in, b_out):
        super(SO3Shortcut, self).__init__()
        assert b_out <= b_in

        if (nfeature_in != nfeature_out) or (b_in != b_out):
            self.conv = SO3Convolution(
                nfeature_in=nfeature_in, nfeature_out=nfeature_out, b_in=b_in, b_out=b_out,
                grid=((0, 0, 0), ))
        else:
            self.conv = None

    def call(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        if self.conv is not None:
            return self.conv(x)
        else:
            return x
