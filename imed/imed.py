from imed.frequency import DCT_ST, FFT_ST
from numpy import sqrt


def transform(
    volume,
    sigma,
    inv=False,
    eps=0,
    method="DCT",
):
    """
    Performs Gaussian standardizing transform (ST) on data volume.
    The function may be used as a image/video preprocessing tool
    to reduce noise; reduce the Euclidean error of small pixel displacements.
    It can also be used as a step towards computing the Image Euclidean Dist.,
    see imed.distance().

    The ST transform can be performed on one or more correlated axes. This is
    specified using using `sigma` parameter.

    The function returns a volume with the same shape as `volume` that has been
    transformed using an ST frequency method, see `method`.

    ST return an altered volume that has the same dtype as `volume`.Currently,
    numpy operations on CPU are not faster when e.g. f32 is chosen over f64.

    Parameters
    ----------
    volume: array_like,
    data volume, for instance a 3D volume consisting of 2D images. May have
    any number of dimensions. ST should only be applied along axes of
    correlated elements. See `sigma` for such configuration.

    sigma: int or array_like,
    Gaussian 'blurring parameter'. Each value leads to a separate IMED
    similarity measure. If `sigma` is an int then that ST is performed
    along all axes of `volume`. Otherwise `sigma` must be array_like with
    dimension 1 and `volume.ndim` number of elements that each specify the
    sigma to be used along the respective axis. When `sigma = 0` is specified,
    that axis is skipped, i.e. no ST/convolution is performed along that these
    axes.

    method: str,
    Method of performing standardizing tranform: 'FFT' or 'DCT' (default).
    Two options provide different edge effects. DCT often provides more natural
    boundary effects in image processing and similar application.

    inv: bool.,
    If true, perform the inverse standardizing transform (deconvolution),
    for example in the context of regression problems. See eps parameter
    for optimal use.

    eps: float,
    If inverse transform is needed for the problem at hand, eps should be a
    small, positive float to avoid catastrophic noise amplification. The need
    increases with the magnitude of sigma. 1e-3 is a good starting point.

    Returns
    -------
    volume_ST: array_like,
    data volume that has undergone the specified standardizing transform.
    if imed scores are needed see imed.euclidean() and imed.distance().
    """
    if method == "DCT":
        volume_ST = DCT_ST(volume, sigma, eps, inv=inv)
    elif method == "FFT":
        volume_ST = FFT_ST(volume, sigma, eps, inv=inv)
    else:
        raise NameError(
            f"`method` must be 'DCT' or 'FFT' not {method}.\n\
            Returning original volume"
        )

    return volume_ST


def euclidean(volume1, volume2, output_dims=0):
    """
    Euclidean distance or pixel-to-pixel dissimilarity of data volumes.
    For a 3D volume/sequence of 2D images, one may require image-to-image
    distance. The optional parameter output_dims specifies the required
    number of dimensions for the output distance. If a single scalar output is
    expected, `output_dims = 0` should be selected (default).

    If volume1, volume2 have dimension `ndim`, summation is performed on the
    last ndim-output_dims axes.

    Parameters
    ----------
    volume1: array_like
    First data volume,; same shape as volume2.

    volume2: array_like
    Second data volume; same shape as volume1.

    euclid_distances: int
    Expected number of dimensions of output distance.

    Returns
    -------
    distances: ndarray
    returns euclidean distance between the two volumes computed
    by summation of the last ndim-output_dims axes of the volumes.
    """
    # check that volume1 and volume2 have same shape
    assert volume1.shape == volume2.shape
    input_dims = volume1.ndim

    # check that output_dims is not larger than volume dimension
    assert output_dims <= input_dims

    sq_deviations = (volume1 - volume2) ** 2

    if output_dims != input_dims:
        sq_deviations = sq_deviations.sum(
            axis=tuple(range(input_dims)[-(input_dims - output_dims) :])
        )

    return sqrt(sq_deviations)


def distance(volume1, volume2, sigma=1, method="DCT", eps=0, output_dims=0):
    """
    Compute the Image Eucliden Distance (IMED) of two volumes
    for a given Gaussian parameter `sigma`.

    Parameters
    ----------
    volume1: array_like
    First data volume,; same shape as volume2.

    volume2: array_like
    Second data volume; same shape as volume1.

    output_dims: int
    Expected number of dimensions of output distance.

    Remaining parameters are defined by imed.transform()

    Returns
    -------
    imeds: array_like,
    Image Euclidean Distances defined according to imed.euclidean()
    """

    volume1_ST = transform(volume1, sigma, False, eps, method)
    volume2_ST = transform(volume2, sigma, False, eps, method)

    imeds = euclidean(volume1_ST, volume2_ST, output_dims)

    return imeds
