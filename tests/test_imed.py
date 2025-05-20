import pytest
import numpy as np
from imed.imed import transform, euclidean, distance

# Helper function to create dummy volumes
def create_volume(shape, dtype=np.float64):
    return np.random.rand(*shape).astype(dtype)

# Tests for transform function
def test_transform_methods():
    volume = create_volume((10, 10))
    sigma = 1.0
    transform(volume, sigma, method='DCT')
    transform(volume, sigma, method='FFT')

def test_transform_invalid_method():
    volume = create_volume((10, 10))
    sigma = 1.0
    with pytest.raises(NameError):
        transform(volume, sigma, method='INVALID')

def test_transform_sigma_scalar():
    volume = create_volume((10, 10))
    transform(volume, 1.0)
    transform(volume, 0)

def test_transform_sigma_array():
    volume = create_volume((10, 10, 5))
    transform(volume, [1.0, 0, 2.0])

def test_transform_inverse():
    volume = create_volume((10, 10))
    sigma = 1.0
    transformed_vol = transform(volume, sigma)
    inverse_transformed_vol = transform(transformed_vol, sigma, inv=True, eps=1e-3)
    # Basic check, transformed back volume should be close to original
    assert np.allclose(volume, inverse_transformed_vol, atol=1e-2)

def test_transform_eps():
    volume = create_volume((10, 10))
    sigma = 1.0
    transform(volume, sigma, inv=True, eps=1e-5)

def test_transform_shapes_types():
    transform(create_volume((5,)), 1.0)
    transform(create_volume((5, 5)), 1.0)
    transform(create_volume((5, 5, 5)), 1.0)
    transform(create_volume((5, 5, 5, 5)), 1.0)
    transform(create_volume((10, 10), dtype=np.float32), 1.0)

# Tests for euclidean function
def test_euclidean_identical_volumes():
    volume1 = create_volume((10, 10))
    volume2 = np.copy(volume1)
    dist = euclidean(volume1, volume2)
    assert np.isclose(dist, 0.0)

def test_euclidean_different_volumes():
    volume1 = create_volume((10, 10))
    volume2 = create_volume((10, 10))
    dist = euclidean(volume1, volume2)
    assert dist > 0.0

def test_euclidean_output_dims():
    volume1 = create_volume((10, 10, 5))
    volume2 = create_volume((10, 10, 5))
    dist_0d = euclidean(volume1, volume2, output_dims=0)
    assert dist_0d.shape == ()
    dist_1d = euclidean(volume1, volume2, output_dims=1)
    assert dist_1d.shape == (10,)
    dist_2d = euclidean(volume1, volume2, output_dims=2)
    assert dist_2d.shape == (10, 10)
    dist_3d = euclidean(volume1, volume2, output_dims=3)
    assert dist_3d.shape == (10, 10, 5)

def test_euclidean_invalid_inputs():
    volume1 = create_volume((10, 10))
    volume2 = create_volume((10, 5))
    with pytest.raises(AssertionError):
        euclidean(volume1, volume2)
    volume3 = create_volume((10, 10, 5))
    with pytest.raises(AssertionError):
        euclidean(volume3, volume3, output_dims=4)

# Tests for distance function
def test_distance_integration():
    volume1 = create_volume((10, 10))
    volume2 = create_volume((10, 10))
    sigma = 1.0
    dist_dct = distance(volume1, volume2, sigma=sigma, method='DCT')
    dist_fft = distance(volume1, volume2, sigma=sigma, method='FFT')
    assert dist_dct.shape == ()
    assert dist_fft.shape == ()
    assert dist_dct >= 0
    assert dist_fft >= 0

def test_distance_parameters():
    volume1 = create_volume((10, 10, 5))
    volume2 = create_volume((10, 10, 5))
    sigma = [1.0, 0, 2.0]
    distance(volume1, volume2, sigma=sigma, method='DCT', eps=1e-4, output_dims=1)