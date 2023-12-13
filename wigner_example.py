# pylint: disable=E1101,R,C

import numpy as np

from keras.datasets import mnist

from notebooks.utils import create_spherical, plot_spherical_img
from wigner_rotations import wigner_rotation

def rotation(x, a, b, c):
    # Get the shape of the input array
    shape = x.shape
    multiples = np.append(np.ones_like(shape), shape[-1])
    x = np.tile(np.expand_dims(x, axis=-1), multiples)
    # Apply the SO(3) rotation, which needs to be defined in NumPy as well
    x = wigner_rotation(x, a, b, c)  
    return x[..., 0]


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    index = 0
    arr = train_X[index]    
    print("projecting {0} data image".format(train_y[index]))
    og_img = create_spherical(arr)[0]
    
    rot_img = rotation(og_img.astype(np.float32), 0, 0, 10)
    rot_img[rot_img<0] = 0
    plot_spherical_img(og_img, rot_img)

    # rot_img2 = rotation(tf.convert_to_tensor(arr, dtype=tf.float32), 0, 0, 2)
    # rot_img2 = rot_img2.numpy()
    # rot_img2[rot_img2<1] = 0
    # plot_spherical_img(arr, rot_img2)
