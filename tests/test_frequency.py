import pytest
import numpy as np
from imed.frequency import DCT_ST, FFT_ST, g12

# Helper function to create dummy images
def create_image(shape, dtype=np.float64):
    return np.random.rand(*shape).astype(dtype)

# Tests for DCT_ST and FFT_ST functions
@pytest.mark.parametrize("transform_func", [DCT_ST, FFT_ST])
def test_frequency_transforms(transform_func):
    img = create_image((10, 10))
    sigma = 1.0
    transform_func(img, sigma)

@pytest.mark.parametrize("transform_func", [DCT_ST, FFT_ST])
def test_frequency_transforms_sigma_scalar(transform_func):
    img = create_image((10, 10))
    transform_func(img, 1.0)
    transform_func(img, 0)

@pytest.mark.parametrize("transform_func", [DCT_ST, FFT_ST])
def test_frequency_transforms_sigma_array(transform_func):
    img = create_image((10, 10, 5))
    transform_func(img, [1.0, 0, 2.0])

@pytest.mark.parametrize("transform_func", [DCT_ST, FFT_ST])
def test_frequency_transforms_inverse(transform_func):
    img = create_image((10, 10))
    sigma = 1.0
    transformed_img = transform_func(img, sigma)
    inverse_transformed_img = transform_func(transformed_img, sigma, inv=True, eps=1e-3)
    # Basic check, transformed back image should be close to original
    assert np.allclose(img, inverse_transformed_img, atol=1e-2)

@pytest.mark.parametrize("transform_func", [DCT_ST, FFT_ST])
def test_frequency_transforms_eps(transform_func):
    img = create_image((10, 10))
    sigma = 1.0
    transform_func(img, sigma, inv=True, eps=1e-5)

@pytest.mark.parametrize("transform_func", [DCT_ST, FFT_ST])
def test_frequency_transforms_shapes_types(transform_func):
    transform_func(create_image((5,)), 1.0)
    transform_func(create_image((5, 5)), 1.0)
    transform_func(create_image((5, 5, 5)), 1.0)
    transform_func(create_image((5, 5, 5, 5)), 1.0)
    transform_func(create_image((10, 10), dtype=np.float32), 1.0)

# Tests for g12 helper function
def test_g12():
    k_d = np.array([0, np.pi/2, np.pi])
    sigma = 1.0
    eps = 0
    result = g12(k_d, sigma, eps)
    # Expected values for sigma=1, eps=0
    expected = np.array([np.exp(0), np.exp(-(np.pi/2)**2/4), np.exp(-np.pi**2/4)])
    assert np.allclose(result, expected)

    eps = 1e-3
    result_eps = g12(k_d, sigma, eps)
    expected_eps = (np.exp(-(k_d ** 2) * sigma ** 2 / 4) + eps) / (1 + eps)
    assert np.allclose(result_eps, expected_eps)

    sigma = 0
    result_sigma_zero = g12(k_d, sigma, eps)
    expected_sigma_zero = (np.exp(0) + eps) / (1 + eps)
    assert np.allclose(result_sigma_zero, expected_sigma_zero)