import os

import numpy as np
import collections

from scipy.linalg import block_diag

from functools import lru_cache

# This code is not very optimized,
# and can never become very efficient because it cannot exploit the sparsity of the J matrix.

# J matrices come from this paper
# Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes
# Didier Pinchon1 and Philip E Hoggan2
# https://iopscience.iop.org/article/10.1088/1751-8113/40/7/011/

# Jd = download('https://github.com/AMLab-Amsterdam/lie_learn/releases/download/v1.0/J_dense_0-278.npy')

base = 'J_dense_0-150.npy'
path = os.path.join(os.path.dirname(__file__), base)
Jd = np.load(path, allow_pickle=True)


def z_rot_mat(angle, l):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    M = np.zeros((2 * l + 1, 2 * l + 1))
    inds = np.arange(0, 2 * l + 1, 1)
    reversed_inds = np.arange(2 * l, -1, -1)
    frequencies = np.arange(l, -l - 1, -1)
    M[inds, reversed_inds] = np.sin(frequencies * angle)
    M[inds, inds] = np.cos(frequencies * angle)
    return M


def rot_mat(alpha, beta, gamma, l, J):
    """
    Compute the representation matrix of a rotation by ZYZ-Euler
    angles (alpha, beta, gamma) in representation l in the basis
    of real spherical harmonics.

    The result is the same as the wignerD_mat function by Johann Goetz,
    when the sign of alpha and gamma is flipped.

    The forementioned function is here:
    https://sites.google.com/site/theodoregoetz/notes/wignerdfunction
    """
    Xa = z_rot_mat(alpha, l)
    Xb = z_rot_mat(beta, l)
    Xc = z_rot_mat(gamma, l)
    return Xa.dot(J).dot(Xb).dot(J).dot(Xc)


def change_of_basis_matrix(l, frm=('complex', 'seismology', 'centered', 'cs'), to=('real', 'quantum', 'centered', 'cs')):
    """
    Compute change-of-basis matrix that takes the 'frm' basis to the 'to' basis.
    Each basis is identified by:
     1) A field (real or complex)
     2) A normalization / phase convention ('seismology', 'quantum', 'nfft', or 'geodesy')
     3) An ordering convention ('centered', 'block')
     4) Whether to use Condon-Shortley phase (-1)^m for m > 0 ('cs', 'nocs')

    Let B = change_of_basis_matrix(l, frm, to).
    Then if Y is a vector in the frm basis, B.dot(Y) represents the same vector in the to basis.

    :param l: the weight (non-negative integer) of the irreducible representation, or an iterable of weights.
    :param frm: a 3-tuple (field, normalization, ordering) indicating the input basis.
    :param to: a 3-tuple (field, normalization, ordering) indicating the output basis.
    :return: a (2 * l + 1, 2 * l + 1) change of basis matrix.
    """
    from_field, from_normalization, from_ordering, from_cs = frm
    to_field, to_normalization, to_ordering, to_cs = to

    if isinstance(l, collections.Iterable):
        blocks = [change_of_basis_matrix(li, frm, to)
                  for li in l]
        return block_diag(*blocks)

    # First, bring us to the centered basis:
    if from_ordering == 'block':
        B = _c2b(l).T
    elif from_ordering == 'centered':
        B = np.eye(2 * l + 1)
    else:
        raise ValueError('Invalid from_order: ' + str(from_ordering))

    # Make sure we're using CS-phase (this should work for both real and complex bases)
    if from_cs == 'nocs':
        m = np.arange(-l, l + 1)
        B = ((-1.) ** (m * (m > 0)))[:, None] * B
    elif from_cs != 'cs':
        raise ValueError('Invalid from_cs: ' + str(from_cs))

    # If needed, change complex to real or real to complex
    # (we know how to do that in the centered, CS-phase bases)
    if from_field != to_field:
        if from_field == 'complex' and to_field == 'real':
            B = _cc2rc(l).dot(B)
        elif from_field == 'real' and to_field == 'complex':
            B = _cc2rc(l).conj().T.dot(B)
        else:
            raise ValueError('Invalid field:' + str(from_field) + ', ' + str(to_field))

    # Set the correct CS phase
    if to_cs == 'nocs':
        # We're in CS phase now, so cancel it:
        m = np.arange(-l, l + 1)
        B = ((-1.) ** (m * (m > 0)))[:, None] * B
    elif to_cs != 'cs':
        raise ValueError('Invalid to_cs: ' + str(to_cs))

    # If needed, change the order from centered:
    if to_ordering == 'block':
        B = _c2b(l).dot(B)
    elif to_ordering != 'centered':
        raise ValueError('Invalid to_ordering:' + str(to_ordering))

    return B


def _c2b(l, full_matrix=True):
    """
    Compute change of basis matrix from the centered basis to
    the Pinchon-Hoggan block basis, in which the Pinchon-Hoggan J matrices
    are brought in block form.

    Let B = c2b(l)
    then B.dot(J_l).dot(B.T) is in block form with 4 blocks,
    as described by PH.
    """
    k = int(l) // 2
    if l % 2 == 0:
        # Permutation as defined by Pinchon-Hoggan for 1-based indices,
        # and l = 2 k
        sigma = np.array([2 * i for i in range(1, 2 * k + 1)]
                         + [2 * i - 1 for i in range(1, 2 * k + 2)])
    else:
        # Permutation as defined by Pinchon-Hoggan for 1-based indices,
        # and l = 2 k + 1
        sigma = np.array([2 * i for i in range(1, 2 * k + 2)]
                         + [2 * i - 1 for i in range(1, 2 * k + 3)])

    if full_matrix:
        # From permutation array sigma, create a permutation matrix B:
        B = np.zeros((2 * l + 1, 2 * l + 1))
        B[np.arange(2 * l + 1), sigma - 1] = 1.
        return B
    else:
        return sigma


def _cc2rc(l):
    """
    Compute change of basis matrix from the complex centered (cc) basis
    to the real centered (rc) basis.

    Let Y be a vector of complex spherical harmonics:
    Y = (Y^{-l}, ..., Y^0, ..., Y^l)^T
    Let S be a vector of real spherical harmonics as defined on the SH wiki page:
    S = (S^{-l}, ..., S^0, ..., S^l)^T
    Let B = cc2rc(l)
    Then S = B.dot(Y)

    B is a complex unitary matrix.

    Formula taken from:
    http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form_2
    """

    B = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            row_ind = m + l
            col_ind = n + l
            if m == 0 and n == 0:
                B[row_ind, col_ind] = np.sqrt(2)
            if m > 0 and m == n:
                B[row_ind, col_ind] = (-1.) ** m
            elif m > 0 and m == -n:
                B[row_ind, col_ind] = 1.
            elif m < 0 and m == n:
                B[row_ind, col_ind] = 1j
            elif m < 0 and m == -n:
                B[row_ind, col_ind] = -1j * ((-1.) ** m)

    return (1.0 / np.sqrt(2)) * B


def wigner_d_matrix(l, beta,
                    field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Compute the Wigner-d matrix of degree l at beta, in the basis defined by
    (field, normalization, order, condon_shortley)

    The Wigner-d matrix of degree l has shape (2l + 1) x (2l + 1).

    :param l: the degree of the Wigner-d function. l >= 0
    :param beta: the argument. 0 <= beta <= pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: d^l_mn(beta) in the chosen basis
    """
    # This returns the d matrix in the (real, quantum-normalized, centered, cs) convention
    d = rot_mat(alpha=0., beta=beta, gamma=0., l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != ('real', 'quantum', 'centered', 'cs'):
        # TODO use change of basis function instead of matrix?
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        d = B.dot(d).dot(BB)

        # The Wigner-d matrices are always real, even in the complex basis
        # (I tested this numerically, and have seen it in several texts)
        # assert np.isclose(np.sum(np.abs(d.imag)), 0.0)
        d = d.real

    return d


def wigner_D_matrix(l, alpha, beta, gamma,
                    field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Evaluate the Wigner-d matrix D^l_mn(alpha, beta, gamma)

    :param l: the degree of the Wigner-d function. l >= 0
    :param alpha: the argument. 0 <= alpha <= 2 pi
    :param beta: the argument. 0 <= beta <= pi
    :param gamma: the argument. 0 <= gamma <= 2 pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: D^l_mn(alpha, beta, gamma) in the chosen basis
    """

    D = rot_mat(alpha=alpha, beta=beta, gamma=gamma, l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != ('real', 'quantum', 'centered', 'cs'):
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        D = B.dot(D).dot(BB)

        if field == 'real':
            # print('WIGNER D IMAG PART:', np.sum(np.abs(D.imag)))
            assert np.isclose(np.sum(np.abs(D.imag)), 0.0)
            D = D.real

    return D


@lru_cache(maxsize=32)
def quadrature_weights(b, grid_type='SOFT'):
    """
    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].

    This grid is:
    alpha = 2 pi i / 2b
    beta = pi (2 j + 1) / 4b
    gamma = 2 pi k / 2b
    where 0 <= i, j, k < 2b are indices
    This grid can be obtained from the function: S3.linspace or S3.meshgrid

    The quadrature weights for this grid are
    w_B(j) = 2/b * sin(pi(2j + 1) / 4b) * sum_{k=0}^{b-1} 1 / (2 k + 1) sin((2j + 1)(2k + 1) pi / 4b)
    This is eq. 23 in [1] and eq. 2.15 in [2].

    [1] SOFT: SO(3) Fourier Transforms
    Peter J. Kostelec and Daniel N. Rockmore

    [2] FFTs on the Rotation Group
    Peter J. Kostelec · Daniel N. Rockmore

    :param b: bandwidth (grid has shape 2b * 2b * 2b)
    :return: w: an array of length 2b containing the quadrature weigths
    """
    if grid_type == 'SOFT':
        k = np.arange(0, b)
        w = np.array([(2. / b) * np.sin(np.pi * (2. * j + 1.) / (4. * b)) *
                      (np.sum((1. / (2 * k + 1))
                              * np.sin((2 * j + 1) * (2 * k + 1)
                                       * np.pi / (4. * b))))
                      for j in range(2 * b)])

        # This is not in the SOFT documentation, but we found that it is necessary to divide by this factor to
        # get correct results.
        w /= 2. * ((2 * b) ** 2)

        # In the SOFT source, they talk about the following weights being used for
        # odd-order transforms. Do not understand this, and the weights used above
        # (defined in the SOFT papers) seems to work.
        # w = np.array([(2. / b) *
        #              (np.sum((1. / (2 * k + 1))
        #                      * np.sin((2 * j + 1) * (2 * k + 1)
        #                               * np.pi / (4. * b))))
        #              for j in range(2 * b)])
        return w
    else:
        raise NotImplementedError
 

@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted):
    dss = _setup_so3_fft(b, nl, weighted)
    dss = dss.astype(np.float32)  # [beta, l * m * n] # pylint: disable=E1102
    return dss


def _setup_so3_fft(b, nl, weighted):
    from wigner_utils import wigner_d_matrix, quadrature_weights
    import numpy as np
    import logging

    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = quadrature_weights(b)
    assert len(w) == len(betas)

    logging.getLogger("trainer").info("Compute Wigner: b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl, nl ** 2)

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', normalization='quantum', order='centered', condon_shortley='cs')
            d = d.reshape(((2 * l + 1) ** 2,))

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            # d # [m * n]
            ds.append(d)
        ds = np.concatenate(ds)  # [l * m * n]
        dss.append(ds)
    dss = np.stack(dss)  # [beta, l * m * n]
    return dss   


def so3_rfft(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha, gamma]
    :return: [l * m * n, ..., complex]
    '''
    b_in = x.shape[-1] // 2
    assert x.shape[-1] == 2 * b_in
    assert x.shape[-2] == 2 * b_in
    assert x.shape[-3] == 2 * b_in
    if b_out is None:
        b_out = b_in
    batch_size = x.shape[:-3]

    x = np.reshape(x, (-1, 2 * b_in, 2 * b_in, 2 * b_in)) # [batch, beta, alpha, gamma]
    '''
    :param x: [batch, beta, alpha, gamma] (nbatch, 2 b_in, 2 b_in, 2 b_in)
    :return: [l * m * n, batch, complex] (b_out (4 b_out**2 - 1) // 3, nbatch, 2)
    '''
    nspec = b_out * (4 * b_out ** 2 - 1) // 3
    nbatch = x.shape[0]

    wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)

    output = np.zeros((nspec, nbatch, 2), dtype=x.dtype)

    fft_x = np.fft.rfft2(x)
    x = np.stack([np.real(fft_x),np.imag(fft_x)], axis=-1)

    if b_in < b_out:
        output = np.zeros_like(output)
    for l in range(b_out):
        s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
        l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null
        
        xx = np.zeros((x.shape[0], x.shape[1], 2 * l + 1, 2 * l + 1, 2))
        xx[:, :, l: l + l1 + 1, l: l + l1 + 1] = x[:, :, :l1 + 1, :l1 + 1]
        
        if l1 > 0:
            xx[:, :, l - l1:l, l: l + l1 + 1] = x[:, :, -l1:, :l1 + 1]
            xx[:, :, l: l + l1 + 1, l - l1:l] = x[:, :, :l1 + 1, -l1:]
            xx[:, :, l - l1:l, l - l1:l] = x[:, :, -l1:, -l1:]

        out = np.einsum("bmn,zbmnc->mnzc", wigner[:, s].reshape(-1, 2 * l + 1, 2 * l + 1), xx)
        output[s] = out.reshape((2 * l + 1) ** 2, -1, 2)
    
    output = np.reshape(output, (-1, *batch_size, 2))  # [l * m * n, ..., complex]
    return output


def so3_rifft(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m * n, ..., complex]
    '''
    assert x.shape[-1] == 2
    nspec = x.shape[0]
    b_in = round((3 / 4 * nspec) ** (1 / 3))
    assert nspec == b_in * (4 * b_in**2 - 1) // 3
    if b_out is None:
        b_out = b_in
    batch_size = x.shape[1:-1]

    x = np.reshape(x, (nspec, -1, 2))  # [l * m * n, batch, complex] (nspec, nbatch, 2)
    
    '''
    :param x: [l * m * n, batch, complex] (b_in (4 b_in**2 - 1) // 3, nbatch, 2)
    :return: [batch, beta, alpha, gamma] (nbatch, 2 b_out, 2 b_out, 2 b_out)
    '''
    nbatch = x.shape[1]

    wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad)  # [beta, l * m * n] (2 * b_out, nspec)

    output = np.zeros((nbatch, 2 * b_out, 2 * b_out, 2 * b_out, 2), dtype=x.dtype)
    # if len(tf.config.experimental.list_physical_devices('GPU')) > 0 and x.dtype == tf.float32:
    #     cuda_kernel = _setup_so3ifft_cuda_kernel(b_in=b_in, b_out=b_out, nbatch=nbatch, real_output=True, device=x.device.index)
    #     cuda_kernel(x, wigner, output)  # [batch, beta, m, n, complex]
    # else:
    for l in range(min(b_in, b_out)):
        start = l * (4 * l**2 - 1) // 3
        end = start + (2 * l + 1)**2
        s = slice(start, end)
        
        out = np.einsum("mnzc,bmn->zbmnc", x[s].reshape(2 * l + 1, 2 * l + 1, -1, 2), wigner[:, s].reshape(-1, 2 * l + 1, 2 * l + 1))
        l1 = min(l, b_out - 1)

        output[:, :, :l1 + 1, :l1 + 1] += out[:, :, l: l + l1 + 1, l: l + l1 + 1]
        if l > 0:
            output[:, :, -l1:, :l1 + 1] += out[:, :, l - l1: l, l: l + l1 + 1]
            output[:, :, :l1 + 1, -l1:] += out[:, :, l: l + l1 + 1, l - l1: l]
            output[:, :, -l1:, -l1:] += out[:, :, l - l1: l, l - l1: l]

    # output = np.fft.ifftn(output.view(np.complex128), axes=[2, 3]) * output.shape[-2] ** 2
    # output = output[..., 0] # [batch, beta, alpha, gamma]

    ifft_output = np.fft.ifft2((output[..., 0] + 1j * output[..., 1]), axes=[2, 3]) * float(output.shape[-2]) ** 2 
    output = np.real(ifft_output)

    output = np.reshape(output, [*batch_size, 2 * b_out, 2 * b_out, 2 * b_out])
    return output
