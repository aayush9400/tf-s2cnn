# pylint: disable=E1101,R,C
import numpy as np
import tensorflow as tf

from keras.datasets import mnist

from notebooks.utils import get_projection_grid, project_2d_on_sphere, plot_spherical_img
from s2cnn.soft.so3_fft import so3_rfft, so3_rifft
from s2cnn.utils.complex import complex_mm


def create_spherical(img, index):
    bandwidth=30
    grid = get_projection_grid(bandwidth=bandwidth)
    print("projecting {0} data image".format(train_y[index]))
    signals = img.reshape(-1, 28, 28).astype(np.float64)
    return project_2d_on_sphere(signals, grid)


def _setup_so3_rotation(b, alpha, beta, gamma):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    Us = [wigner_D_matrix(l, alpha, beta, gamma,
                          field='complex', normalization='quantum', 
                          order='centered', condon_shortley='cs')
          for l in range(b)]
          
    Us = [Us[l].astype(np.complex64).view(np.float32).reshape((2 * l + 1, 2 * l + 1, 2)) for l in range(b)]

    return Us


def so3_rotation(x, alpha, beta, gamma):
    '''
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    '''
    b = x.shape[-1] // 2
    x_size = x.shape
    Us = _setup_so3_rotation(b, alpha, beta, gamma)
    # fourier transform
    x = so3_rfft(x, b_out=None)  # [l * m * n, ..., complex]
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
    z = so3_rifft(Fz, b_out=None)
    z = tf.reshape(z, x_size)
    return z


def rotation(x, a, b, c):
    # Get the shape of the input array
    shape = x.shape
    multiples = np.append(np.ones_like(shape), shape[-1])
    x = np.tile(np.expand_dims(x, axis=-1), multiples)
    # Apply the SO(3) rotation, which needs to be defined in NumPy as well
    x = so3_rotation(x, a, b, c)  
    return x[..., 0]

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    index = 2
    arr = train_X[index]    

    og_img = create_spherical(arr, index)
    
    rot_img = rotation(tf.convert_to_tensor(og_img[0], dtype=tf.float32), 0, 2, 4)
    rot_img2 = rotation(tf.convert_to_tensor(arr, dtype=tf.float32), 0, 0, 2)
    rot_img = rot_img.numpy()
    rot_img[rot_img<0] = 0

    rot_img2 = rot_img2.numpy()
    # rot_img2[rot_img2<1] = 0
    print(rot_img.min(), rot_img.max())


    plot_spherical_img(og_img[0], rot_img)
    # plot_spherical_img(arr, rot_img2)
