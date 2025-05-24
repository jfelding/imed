import warnings
import numpy as np
from numpy import sqrt, exp, pi, dot, zeros, arange, maximum


def fullMat_ST(imgs, sigma, inverse=False):
    """
    This is created by Niklas Heim / James Avery; bugfixed by Jacob Felding
    This method is present for historical reasons and is highly inefficient
    for large inputs due to the creation and manipulation of an (M*N) x (M*N) matrix.
    2D transform only.
    The `sepMat_ST` method or DCT/FFT frequency methods are strongly recommended
    over this one for better performance and lower memory usage.
    """
    warnings.warn(
        "fullMat_ST is highly inefficient and deprecated. Use sepMat_ST or frequency methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Purposefully not M, N
    # sigma check
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, there should be 2 for spatial transform
        print("This method allows only single sigma values. Using 1st one")
        sigma = sigma[0]
    if len(imgs.shape) == 3:
        N, M = imgs.shape[1:]
    else:
        N, M = imgs.shape
        imgs = np.expand_dims(imgs, axis=0)

    X = arange(M, dtype=imgs.dtype)
    Y = arange(N, dtype=imgs.dtype)

    P = (X[None, :, None, None] - X[None, None, None, :]) ** 2 + (
        Y[:, None, None, None] - Y[None, None, :, None]
    ) ** 2

    G = 1 / (2 * pi * sigma ** 2) * exp(-P / (2 * sigma ** 2))

    # Construct NM x NM G matrix
    G = G.reshape((M * N, M * N))

    # Use eigenvalue decomposition to construct G^1/2
    (w, V) = np.linalg.eigh(G)
    # G^(1/2) has eigenvalues that are the square root of eigenvalues of G
    w = maximum(zeros(len(w), dtype=imgs.dtype), w)
    s = sqrt(w)

    # Eigenvalue decomposition
    # G_sqrt  = dot(V, s[:,None]*V.T)

    if inverse:
        VTS = -s[:, None] * V.T
    else:
        VTS = s[:, None] * V.T

    imgs = np.array([dot(V, dot(VTS, z.reshape(-1))) for z in imgs])
    return imgs.reshape(-1, N, M)


def sepMat_ST(imgs, sigma, inverse=False):
    """
    Implements 2D ST of image sequence using tensor product.
    Much faster, and less memory intensive than fullMat_ST, but
    not as fast as DCT/FFT frequency methods. Only available as 2D transform
    """

    # Make sigma 2-dimensional if not already
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # Should be 2 sigmas for spatial transform
        assert len(sigma) == 2
    else:
        # if sigma is a scalar, it will be used for all axes
        sigma = np.ones(2) * sigma

    if len(imgs.shape) == 3:
        N, M = imgs.shape[1:]
    else:
        N, M = imgs.shape

    # Create 1D arrays corresponding to image dimensions
    X = arange(M, dtype=imgs.dtype)
    Y = arange(N, dtype=imgs.dtype)

    # Create 2D arrays where P_x(i,j) denotes (x_i-x_j)**2
    P_x = (X[None, :] - X[:, None]) ** 2
    P_y = (Y[None, :] - Y[:, None]) ** 2

    # Create G decompositions such that G = np.kron(G_x,G_y)
    # This way we store NxN, MxM matrices, not NMxNM
    G_x = 1 / (sqrt((2 * pi)) * sigma[0]) * exp(-P_x / (2 * sigma[0] ** 2))
    G_y = 1 / (sqrt((2 * pi)) * sigma[0]) * exp(-P_y / (2 * sigma[1] ** 2))

    # Determine eigenvectors and of G using that G = np.kron(G_x,G_y)
    # The matrices are symmetric since (x_i-x_j)**2 = (x_j-x_i)**2
    evals_Gx, evecs_Gx = np.linalg.eigh(G_x)
    evals_Gy, evecs_Gy = np.linalg.eigh(G_y)

    # For robustness to negative eigenvalues that should not arise as
    # G is positive definte, but may for numerical reasons
    evals_Gx = maximum(evals_Gx, zeros(M, dtype=imgs.dtype))
    evals_Gy = maximum(evals_Gy, zeros(N, dtype=imgs.dtype))

    # For the Standardizing Transfrom G^(1/2)x (which blurs image x)
    # we need matrix G^(1/2) Such that G = G^(1/2)G^(1/2)
    # Again decomposition allows G^(1/2) = np.kron(G_sqrt_x,G_sqrt_y)
    # Below, use eigendecomposition of G to construct G_sqrt_x, G_sqrt_y

    if inverse:
        G_sqrt_x = dot(evecs_Gx, -sqrt(evals_Gx)[:, None] * evecs_Gx.T)
        G_sqrt_y = dot(evecs_Gy, -sqrt(evals_Gy)[:, None] * evecs_Gy.T)
    else:
        G_sqrt_x = dot(evecs_Gx, sqrt(evals_Gx)[:, None] * evecs_Gx.T)
        G_sqrt_y = dot(evecs_Gy, sqrt(evals_Gy)[:, None] * evecs_Gy.T)

    return [dot(G_sqrt_x, dot(z, G_sqrt_y)) for z in imgs]
