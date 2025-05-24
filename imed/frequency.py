import numpy as np
from numpy import pi, prod, exp, ones, ndarray, asarray
from scipy.fft import dct, idct
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq

NA = np.newaxis


def axis_split(A, axis):
    """helper function to avoid unnecessary copy/numpy.repeat"""
    (Am, Ai, An) = (
        int(prod(A.shape[:axis])),
        A.shape[axis],
        int(prod(A.shape[axis + 1 :])),
    )
    return (Am, Ai, An)


def g12(k_d, sigma, eps):
    """
    Square root of fourier transformed Gaussian.
    Used in both FFT and DCT context where sampled frequencies k_d
    are configured outside of this function.
    when eps is small, positive constant, inverse ST is feasible.
    """
    return (exp(-(k_d ** 2) * sigma ** 2 / 4) + eps) / (1 + eps)


def DCT_1dim(Img, sigma, axis=0, eps=0, inv=False):
    """Perform ST symmetric convolution along axis `axis`"""
    # Sample sqrt of Gaussian in k-space
    k_d = np.linspace(0, pi, Img.shape[axis], dtype=Img.dtype)
    filter = g12(k_d, sigma, eps)

    # Transform to k-space
    img_dct = dct(Img, axis=axis, type=1).reshape(axis_split(Img, axis))

    if inv:
        Img_folded_k = img_dct / filter[NA, :, NA]
    else:
        Img_folded_k = img_dct * filter[NA, :, NA]

    Img_folded = idct(Img_folded_k, axis=1, type=1)

    return Img_folded.reshape(Img.shape)


def FFT_1dim(Img, sigma, axis=0, eps=0, inv=False):
    """Perform ST periodic convolution along axis `axis`"""

    # Transform to k-space
    img_fft = rfftn(Img)
    fft_shape = img_fft.shape
    img_fft = img_fft.reshape(axis_split(img_fft, axis))

    # if last axis, need other k definition for rfft
    if axis == Img.ndim - 1:
        k_d = rfftfreq(2 * fft_shape[axis] - 1) * 2 * pi
        # ensure correct dtype
        k_d = asarray(k_d, dtype=Img.dtype)

    else:
        k_d = fftfreq(fft_shape[axis]) * 2 * pi
        # ensure correct dtype
        k_d = asarray(k_d, dtype=Img.dtype)

    # Gaussian in k-space
    filter = g12(k_d, sigma, eps)

    if inv:
        Img_folded_k = img_fft / filter[NA, :, NA]
    else:
        Img_folded_k = img_fft * filter[NA, :, NA]

    Img_folded = irfftn(Img_folded_k.reshape(fft_shape), s=Img.shape)

    return Img_folded.reshape(Img.shape)


def DCT_ST(imgs, sigma, eps=0, inv=False):
    # automatic d-dimensional standardizing transform
    # via DCT, i.e. symmetric boundary conditions
    # eps is an optional constant added to the OTF to reduce
    # noise amplification when deconvolving
    shape = imgs.shape
    dims = len(shape)

    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        # if sigma is a scalar, it will be used for all axes
        sigma = ones(dims) * sigma

    # do convolution, axis by axis
    for axis in range(dims):
        if sigma[axis] == 0:
            # convolution has no effect
            continue

        if shape[axis] < 2:
            # cant do convolution along this axis
            continue

        imgs = DCT_1dim(imgs, sigma[axis], axis, eps, inv)
    return imgs


def DCT_by_FFT_ST(imgs, sigma, eps=0, inv=False):
    """
    Automatic n-dimensional standardizing transform
    with output like DCT_ST, but using FFT algorithm (periodic convolution).
    This method involves mirroring and concatenating arrays, which is
    memory-intensive for large inputs. For general DCT-based transforms,
    `DCT_ST` is more memory and CPU efficient as it directly uses
    optimized DCT implementations.
    eps is an optional constant added to the OTF to reduce
    noise amplification when inverse ST is needed
    """
    import warnings
    warnings.warn(
        "DCT_by_FFT_ST is memory-intensive for large inputs due to temporary array creation. "
        "Consider using DCT_ST for better performance and memory efficiency.",
        UserWarning,
        stacklevel=2
    )

    orig_shape = imgs.shape
    dims = len(orig_shape)
    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        # if sigma is a scalar, it will be used for all axes
        sigma = ones(dims) * sigma

    for axis in range(dims):
        if sigma[axis] == 0:
            # convolution has no effect
            continue
        if orig_shape[axis] < 3:
            print(f"Skipping axis={axis} with size {orig_shape[axis]}")
            # cant do convolution along this axis
            continue

        # mirror image along axis `axis`
        imgs_reverse = np.flip(imgs, axis=axis)
        # for DCT equivalence
        imgs_reverse = imgs_reverse.take(
            indices=range(1, imgs_reverse.shape[axis] - 1), axis=axis
        )
        imgs = np.concatenate((imgs, imgs_reverse), axis=axis)
        imgs = FFT_1dim(imgs, sigma[axis], axis, eps, inv)

        # Cut to original shape before moving on to other axis
        imgs = imgs.take(indices=range(orig_shape[axis]), axis=axis)

    return imgs


def FFT_ST(imgs, sigma, eps=0, inv=False):
    """
    n-dimensional standardizing transform via FFT.
    Uses per-axis mirroring to reduce edge discontinuities
    eps is an optional constant added to the OTF to reduce
    noise amplification when inverse ST is needed
    """
    orig_shape = imgs.shape
    dims = len(orig_shape)

    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        # if sigma is a scalar it will be used for all axes
        sigma = ones(dims) * sigma

    for axis in range(dims):
        if sigma[axis] == 0:
            # convolution has no effect
            continue

        if orig_shape[axis] < 2:
            # can't do convolution along this axis
            continue

        imgs = FFT_1dim(imgs, sigma[axis], axis, eps, inv)

    return imgs
