import numpy as np
from keras.datasets import mnist
from scipy.spatial.transform import Rotation as R

from fury import actor, window
from fury.lib import ImageData, FloatArray, Texture, numpy_support
from fury.primitive import prim_sphere
from fury.utils import (
    get_polydata_normals,
    set_polydata_tcoords,
    update_polydata_normals,
)

NORTHPOLE_EPSILON = 1e-3
from dipy.reconst.shm import wigner_rotation


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def spherical_to_cartesian(spherical_coords, r=1):
    cartesian_coords = []
    for theta, phi in spherical_coords:
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        cartesian_coords.append((x, y, z))
    return np.array(cartesian_coords)


def plot_spherical_img(img1, img2=None, points=None, rotations=None, separation=2):
    def np_array_to_vtk_img(data):
        grid = ImageData()
        grid.SetDimensions(data.shape[1], data.shape[0], 1)
        nd = data.shape[-1] if data.ndim == 3 else 1
        vtkarr = numpy_support.numpy_to_vtk(
            np.flip(data.swapaxes(0, 1), axis=1).reshape((-1, nd), order="F")
        )
        vtkarr.SetName("Image")
        grid.GetPointData().AddArray(vtkarr)
        grid.GetPointData().SetActiveScalars("Image")
        grid.GetPointData().Update()
        return grid

    def create_texture_actor(img, position):
        arr = np.expand_dims(img, -1)
        grid = np_array_to_vtk_img(arr.astype(np.uint8))

        center = np.array([position])
        vertices, faces = prim_sphere(name='repulsion724')
        texture_actor = actor.sphere(center, (1, 1, 1), vertices=vertices, faces=faces, use_primitive=True)

        actor_pd = texture_actor.GetMapper().GetInput()
        update_polydata_normals(actor_pd)
        normals = get_polydata_normals(actor_pd)

        u_vals = np.arctan2(normals[:, 0], normals[:, 2]) / (2 * np.pi) + .5
        v_vals = normals[:, 1] * .5 + .5
        num_pnts = normals.shape[0]

        t_coords = FloatArray()
        t_coords.SetNumberOfComponents(2)
        t_coords.SetNumberOfTuples(num_pnts)

        for i in range(num_pnts):
            u = u_vals[i]
            v = v_vals[i]
            tc = [u, v]
            t_coords.SetTuple(i, tc)

        set_polydata_tcoords(actor_pd, t_coords)

        texture = Texture()
        texture.SetInputDataObject(grid)
        texture.Update()

        texture_actor.SetTexture(texture)

        return texture_actor

    scene = window.Scene()
    scene.background((1, 1, 1))

    texture_actor1 = create_texture_actor(img1, [-separation / 2, 0, 0])
    scene.add(texture_actor1)

    if img2 is not None:
        texture_actor2 = create_texture_actor(img2, [separation / 2, 0, 0])
        scene.add(texture_actor2)
        
    if points:
        cartesian_points = spherical_to_cartesian(points) - np.array([[separation / 2, 0, 0]])
        point_actor = actor.point(cartesian_points, colors=(1, 0, 0), point_radius=0.05)  # Red points
        scene.add(point_actor)
    elif rotations:
        points = []
        for rotation in rotations:
            r = R.from_euler('zyx', rotation, degrees=False)
            point = r.apply([1, 0, 0])
            points.append(point)

        points = np.array(points) - np.array([[separation / 2, 0, 0]])
        point_actor = actor.point(points, colors=(1, 0, 0), point_radius=0.05)  # Red points
        scene.add(point_actor)

    window.show(scene)

    return True


def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'equidistribution':
        raise NotImplementedError('Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.')
    else:
        raise ValueError('Unknown grid_type:' + grid_type)
    return beta, alpha


def get_projection_grid(bandwidth, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = np.meshgrid(*linspace(bandwidth, grid_type), indexing='ij')
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def project_2d_on_sphere(signal, grid, projection_origin=None):
    ''' '''
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry)

    # ensure that only south hemisphere gets projected
    sample *= (grid[2] <= 1).astype(np.float64)

    # rescale signal to [0,1]
    sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    sample = (sample - sample_min) / (sample_max - sample_min)
    sample *= 255
    sample = sample.astype(np.uint8)

    return sample


def create_spherical(img):
    bandwidth=30
    grid = get_projection_grid(bandwidth=bandwidth)
    signals = img.reshape(-1, 28, 28).astype(np.float64)
    return project_2d_on_sphere(signals, grid)


def rotation(x, a, b, c):
    shape = x.shape
    multiples = np.append(np.ones_like(shape), shape[-1])
    x = np.tile(np.expand_dims(x, axis=-1), multiples)
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