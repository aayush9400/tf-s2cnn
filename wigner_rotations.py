import numpy as np
from wigner_utils import so3_rfft, so3_rifft

def complex_mm(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [i, k, complex] (M, K, 2)
    :param y: [k, j, complex] (K, N, 2)
    :return:  [i, j, complex] (M, N, 2)
    '''
    xr = x[:, :, 0]
    xi = x[:, :, 1]

    yr = y[:, :, 0]
    yi = y[:, :, 1]

    if not conj_x and not conj_y:
        zr = np.matmul(xr, yr) - np.matmul(xi, yi)
        zi = np.matmul(xr, yi) + np.matmul(xi, yr)
    if conj_x and not conj_y:
        zr = np.matmul(xr, yr) + np.matmul(xi, yi)
        zi = np.matmul(xr, yi) - np.matmul(xi, yr)
    if not conj_x and conj_y:
        zr = np.matmul(xr, yr) + np.matmul(xi, yi)
        zi = np.matmul(xi, yr) - np.matmul(xr, yi)
    if conj_x and conj_y:
        zr = np.matmul(xr, yr) - np.matmul(xi, yi)
        zi = - np.matmul(xr, yi) - np.matmul(xi, yr)

    return np.stack((zr, zi), axis=2)


def _setup_so3_rotation(b, alpha, beta, gamma):
    from wigner_utils import wigner_D_matrix

    Us = [wigner_D_matrix(l, alpha, beta, gamma,
                          field='complex', normalization='quantum', order='centered', condon_shortley='cs')
          for l in range(b)]
    Us = [Us[l].astype(np.complex64).view(np.float32).reshape((2 * l + 1, 2 * l + 1, 2)) for l in range(b)]
    return Us


def wigner_rotation(x, alpha, beta, gamma):
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
        Fx = np.reshape(Fx, (L, -1, 2))  # [m, n * batch, complex]

        U = np.reshape(Us[l], (L, L, 2))  # [m, n, complex]

        Fz = complex_mm(U, Fx, conj_x=True)  # [m, n * batch, complex]
        Fz = np.reshape(Fz, (size, -1, 2))  # [m * n, batch, complex]
        
        Fz_list.append(Fz)
        begin += size
    Fz = np.concatenate(Fz_list, axis=0)  # [l * m * n, batch, complex]
    z = so3_rifft(Fz, b_out=None)
    z = np.reshape(z, x_size)
    return z
