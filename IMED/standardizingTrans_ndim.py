import numpy as np
from scipy.fft import dct, idct
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq

NA = np.newaxis


def axis_split(A, axis):
    """helper function to avoid unnecessary copy/numpy.repeat"""
    (Am, Ai, An) = (
        int(np.prod(A.shape[:axis])),
        A.shape[axis],
        int(np.prod(A.shape[axis + 1 :])),
    )
    return (Am, Ai, An)


def DCT_1dim(Img, sigma, axis=0, eps=0, inv=False):
    """Perform ST symmetric convolution along axis `axis`"""
    # Sample sqrt of Gaussian in k-space
    k_d = np.linspace(0, np.pi, Img.shape[axis], dtype=Img.dtype)
    g12_dct = (np.exp(-(k_d ** 2) * sigma ** 2 / 4) + eps) / (1 + eps)

    # Transform to k-space
    img_dct = dct(Img, axis=axis, type=1).reshape(axis_split(Img, axis))

    if inv:
        Img_folded_k = img_dct / g12_dct[NA, :, NA]
    else:
        Img_folded_k = img_dct * g12_dct[NA, :, NA]

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
        k_d = rfftfreq(2 * fft_shape[axis] - 1) * 2 * np.pi
        # ensure correct dtype
        k_d = np.asarray(k_d, dtype=Img.dtype)

    else:
        k_d = fftfreq(fft_shape[axis]) * 2 * np.pi
        # ensure correct dtype
        k_d = np.asarray(k_d, dtype=Img.dtype)

    # Gaussian in k-space
    g12_fft = (np.exp(-(k_d ** 2) * sigma ** 2 / 4) + eps) / (1 + eps)

    if inv:
        Img_folded_k = img_fft / g12_fft[NA, :, NA]
    else:
        Img_folded_k = img_fft * g12_fft[NA, :, NA]

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
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        # if sigma is a scalar, it will be used for all axes
        sigma = np.ones(dims) * sigma

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
    eps is an optional constant added to the OTF to reduce
    noise amplification when inverse ST is needed
    """

    orig_shape = imgs.shape
    dims = len(orig_shape)
    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        # if sigma is a scalar, it will be used for all axes
        sigma = np.ones(dims) * sigma

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
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        # if sigma is a scalar it will be used for all axes
        sigma = np.ones(dims) * sigma

    for axis in range(dims):
        if sigma[axis] == 0:
            # convolution has no effect
            continue

        if orig_shape[axis] < 2:
            # can't do convolution along this axis
            continue

        imgs = FFT_1dim(imgs, sigma[axis], axis, eps, inv)

    return imgs
