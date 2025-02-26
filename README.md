# Spherical CNNs
## Equivariant CNNs for the sphere and SO(3) implemented in TensorFlow

![Equivariance](https://github.com/aayush9400/tf-s2cnn/blob/master/examples/equivariance_plot/fig.jpeg)

## Overview
This library contains a TensorFlow implementation of the rotation equivariant CNNs for spherical signals (e.g. omnidirectional images, signals on the globe) as presented in [[1]](https://arxiv.org/abs/1801.10130). Equivariant networks for the plane are available [here](https://github.com/tscohen/GrouPy).

## Dependencies

* __TensorFlow__: https://www.tensorflow.org/ (>= 2.0)
* __cython__: https://github.com/cython/cython (<= 3.0)
* __lie_learn__: https://github.com/kalekundert/lie_learn@fix-install

(commands to install all the dependencies on a new environment)
```bash
# tf-s2cnn deps
python -m pip install --upgrade pip wheel cython<3.0
pip install git+https://github.com/kalekundert/lie_learn@fix-install

pip install tensorflow==2.14.0  # get correct command line at https://www.tensorflow.org/
pip install tqdm

# update venv\lib\site-packages\lie_learn\representations\SO3\wigner_d.py
import collections
collections.Iterable = collections.abc.Iterable

```
<!---
# shrec17 example dep
conda install -c anaconda scipy  
conda install -c conda-forge rtree shapely  
conda install -c conda-forge pyembree  
pip install "trimesh[easy]"
-->

## Usage
Please have a look at the [examples](examples).

Please cite [[1]](https://arxiv.org/abs/1801.10130) in your work when using this library in your experiments.


## Design choices for Spherical CNN Architectures

Spherical CNNs come with different choices of grids and grid hyperparameters which are on the first look not obviously related to those of conventional CNNs.
The `s2_near_identity_grid` and `so3_near_identity_grid` are the preferred choices since they correspond to spatially localized kernels, defined at the north pole and rotated over the sphere via the action of SO(3).
In contrast, `s2_equatorial_grid` and `so3_equatorial_grid` define line-like (or ring-like) kernels around the equator.

To clarify the possible parameter choices for `s2_near_identity_grid`:
#### max_beta:
Adapts the size of the kernel as the angle measured from the north pole.
Conventional CNNs on flat space usually use a fixed kernel size but pool the signal spatially.
This spatial pooling gives the kernels in later layers an effectively increased field of view.
One can emulate a pooling by a factor of 2 in spherical CNNs by decreasing the signal bandwidth by 2 and increasing `max_beta` by 2.
#### n_beta:
Number of rings of the kernel around the equator, equally spaced in
[&beta;=0, &beta;=`max_beta`].
The choice `n_beta=1` corresponds to a small 3x3 kernel in `conv2d` since in both cases the resulting kernel consists of one central pixel and one ring around the center.
#### n_alpha:
Gives the number of learned parameters of the rings around the pole.
These values are per default equally spaced on the azimuth.
A sensible number of values depends on the bandwidth and `max_beta` since a higher resolution or spatial extent allows to sample more fine kernels without producing aliased results.
In practice this value is typically set to a constant, low value like 6 or 8.
A reduced bandwidth of the signal is thereby counteracted by an increased `max_beta` to emulate spatial pooling.

The `so3_near_identity_grid` has two additional parameters `max_gamma` and `n_gamma`.
SO(3) can be seen as a (principal) fiber bundle SO(3)&rarr;S&sup2; with the sphere S&sup2; as base space and fiber SO(2) attached to each point.
The additional parameters control the grid on the fiber in the following way:
#### max_gamma:
The kernel spans over the fiber SO(2) between &gamma;&isin;[0, `max_gamma`].
The fiber SO(2) encodes the kernel responses for every sampled orientation at a given position on the sphere.
Setting `max_gamma`&#8808;2&pi; results in the kernel not seeing the responses of all kernel orientations simultaneously and is in general unfavored.
Steerable CNNs [[3]](https://arxiv.org/abs/1803.10743) usually always use `max_gamma`=2&pi;.
#### n_gamma:
Number of learned parameters on the fiber.
Typically set equal to `n_alpha`, i.e. to a low value like 6 or 8.

See the deep model of the MNIST example for an example of how to adapt these parameters over layers.


## References

[1] Taco S. Cohen, Mario Geiger, Jonas Köhler, Max Welling,
[Spherical CNNs](https://arxiv.org/abs/1801.10130).
International Conference on Learning Representations (ICLR), 2018.

[2] Taco S. Cohen, Mario Geiger, Jonas Köhler, Max Welling,
[Convolutional Networks for Spherical Signals](https://arxiv.org/abs/1709.04893).
ICML Workshop on Principled Approaches to Deep Learning, 2017.

[3] Taco S. Cohen, Mario Geiger, Maurice Weiler,
[Intertwiners between Induced Representations (with applications to the theory of equivariant neural networks)](https://arxiv.org/abs/1803.10743),
ArXiv preprint 1803.10743, 2018.
