# IMage Euclidean Distance (IMED)

Matrix and Frequency implementation based on https://link.springer.com/content/pdf/10.1186/s40535-015-0014-6.pdf

The Image Euclidean Distance is found by transforming images or other continous data using a convolution operation, then taking the standard pixel-wise Euclidean distance between the image. This transform is referred to as the 'Standardizing Transform', or ST. 

This package contains five implementations of the standardizing transform. Two of them ('full', 'sep') are matrix methods that apply linear convolution, while the remainder are frequency/DFT methods ('fft', 'dct', 'dct_by_fft'). the 'fft' performs the transform using circular convolution, while 'dct' and 'dct_by_fft' give identical results and apply symmetric convolution.

The most natural boundaries are often obtained using 'dct' and 'dct_by_fft' methods. The linear convolution methods tend to underestimate the boundaries, as they correspond to zero-padding the image berfore applying a certain Gaussian filter. 'fft' tends to give periodic boundary effects.

The frequency methods apply an n-dimensional transform and so can be used for continous signals with any number of dimensions. The matrix methods are 2D only.

The 'full' method is the original method, and is only here for completeness. It consumes a lot of memory, and is very slow. Its use is not recommended.

## Getting Started with the Standardizing Transform
**Installation**:
Install the latest release from pypi:

    pip install IMED

To get started, IMED.ST_all contains a wrapper function standardizingTrans(imgs,sigma,method,eps=0,inverse=False)
Here is the doc:

    Takes n-dimensional data and returns the n-dimensional Standardized Transform.
    Methods 'full' and 'sep' are 2D methods only.
    
    Parameters:
    * imgs is a signal 
    * sigma (float)/array-like determines the zero-mean Gaussian that defines the IMED matrix G - not G^(1/2).
      If sigma is array-like it should contain the same number of values as the number of dimensions of imgs.
      
    * eps (float) is an optional small parameter to offset the Gaussian so that it is always numerically non-zero. 
    This can allow deconvolution without significant noise amplification.
    * method (string) is the method used to perform the standardizing transform. Choose between:
     1. **'full':** Full Dense Matrix $z_{ST}= G^{1/2}z$ using eigenvalue decomposition
     2. **'sep'** Separated Dense Matrices $z_{ST}= G_x^{1/2}z G_y^{1/2}$ using eigenvalue decomposition 
     3. **'fft'**: Performs circular convolution using discrete fourier transforms of image and Gaussian 
     without enforcing symmetric boundary conditions
     4. **'dct_by_fft'**: Performs circular convolution using discrete fourier transforms of mirrored image and 
     Gaussian to ensure symmetric boundary conditions and reduce edge effects from 'Gibbs-like phenomenon'
     5. **'dct'**: Performs symmetric convolution using discrete cosine transform of image, which is identical to
     6. the 'dct_by_fft  method, but should be more efficient
    """

## Parallelization of frequency methods and backend change
The frequency methods of the standardizing transform ('fft', 'dct' and 'dct_by_fft') are implemented using scipy.fft.
By chopping up the transforms into smaller chunks, scipy supports parallelization by specifying the _workers_ environment variable:
    
    from scipy import fft
    with fft.set_workers(-1):
        standardizingTrans(imgs,sigma,method,eps=0,inverse=False)
        
When the number of workers is set to a negative integer (like above), the number of workers is set to os.cpu_count().

Scipy also supports computations using another backend. For example, we can use pyFFTW as a backend like so: 
     
     from scipy import fft
     import pyfftw
     with fft.set_backend(pyfftw.interfaces.scipy_fft):
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        imgs_ST = standardizingTrans(imgs,sigma=(10.,20.),method='DCT',eps=0,inverse=False)
        
## Inverse Transforms
In principle, the frequency methods allow simple inverse ST by inverse filtering, and this can be triggered using the *inverse=True* flag. However, in the present of any noise, catastrophic noise amplification can occur, especially since the filter used in this transform is Gaussian and tends towards zero. The effect will increase with sigma. If an inverse transformation is necessary, it is recommended to use the **eps** parameter to add a small constant like 1e-5 to the filter in frequency space, i.e. the optical transfer function. The **eps** parameter is also available for (forward) standardizing transforms.
