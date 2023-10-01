# pylint: disable=C,R,E1101,E1102
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import tensorflow as tf
from s2cnn import S2Convolution, SO3Convolution, so3_rotation
from s2cnn import s2_near_identity_grid, so3_near_identity_grid


def s2_rotation(x, a, b, c):
    shape = tf.shape(x)

    x = tf.expand_dims(x, axis=-1)
    multiples = tf.concat([tf.ones_like(shape, dtype=tf.int32), [shape[-1]]], axis=0)
    x = tf.tile(x, multiples)

    x = so3_rotation(x, a, b, c) 
    return x[..., 0]


def plot(x, text, normalize=False):
    assert x.shape[0] == 1
    assert x.shape[1] in [1, 3]
    x = x[0]
    if len(x.shape) == 4:
        x = x[..., 0]

    nch = x.shape[0]
    is_rgb = (nch == 3)

    if normalize:
        x -= tf.reduce_mean(tf.reshape(x, (nch, -1)), axis=-1, keepdims=True)
        x = 0.4 * x / tf.math.reduce_std(tf.reshape(x, (nch, -1)), axis=-1, keepdims=True)

    if tf.executing_eagerly():
        x = x.numpy()  # Convert to numpy array if in eager execution mode
    else:
        with tf.Session() as sess:
            x = x.eval()  # Evaluate the tensor to get a numpy array if in graph mode

    x = np.transpose(x, (1, 2, 0)).clip(0, 1)

    print(x.shape)
    if is_rgb:
        plt.imshow(x)
    else:
        plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis("off")

    plt.text(0.5, 0.5, text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             color='white', fontsize=20)


def main():
    device = "/gpu:0" if tf.config.experimental.list_physical_devices("GPU") else "/cpu:0"

    # load image
    x = imread("earth128.jpg").astype(np.float32).transpose((2, 0, 1)) / 255
    b = 64
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.reshape(x, (1, 3, 2*b, 2*b))

    # equivariant transformation
    s2_grid = s2_near_identity_grid(max_beta=0.2, n_alpha=12, n_beta=1)
    s2_conv = S2Convolution(3, 50, b_in=b, b_out=b, grid=s2_grid)

    so3_grid = so3_near_identity_grid(max_beta=0.2, n_alpha=12, n_beta=1)
    so3_conv = SO3Convolution(50, 1, b_in=b, b_out=b, grid=so3_grid)

    def phi(x):
        x = s2_conv(x)
        x = tf.math.softplus(x)
        x = so3_conv(x)
        return x

    # test equivariance
    abc = (0.5, 1, 0)  # rotation angles

    y1 = phi(s2_rotation(x, *abc))
    y2 = so3_rotation(phi(x), *abc)

    difference_std = tf.math.reduce_std(y1 - y2)
    y1_std = tf.math.reduce_std(y1)

    # Print the standard deviations
    print(difference_std.numpy(), y1_std.numpy())

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plot(x, "x : signal on the sphere")

    plt.subplot(2, 3, 2)
    plot(phi(x), "phi(x) : convolutions", True)

    plt.subplot(2, 3, 3)
    plot(so3_rotation(phi(x), *abc), "R(phi(x))", True)

    plt.subplot(2, 3, 4)
    plot(s2_rotation(x, *abc), "R(x) : rotation using fft")

    plt.subplot(2, 3, 5)
    plot(phi(s2_rotation(x, *abc)), "phi(R(x))", True)

    plt.tight_layout()
    plt.savefig("fig.jpeg")


if __name__ == "__main__":
    main()
