# IMage Euclidean Distance (IMED)
The Image Euclidean Distance is found by transforming images or other continous data using a convolution operation, then taking the standard pixel-wise Euclidean distance between the image. This transform is referred to as the 'Standardizing Transform', or ST. This package contains five implementations of the standardizing transform. Two of them ('full', 'sep') are matrix methods that apply linear convolution, while the remainder are frequency/DFT methods ('fft', 'dct', 'dct_by_fft'). the 'fft' performs the transform using circular convolution, while 'dct' and 'dct_by_fft' give identical results and apply symmetric convolution.

The most natural boundaries are often obtained using 'dct' and 'dct_by_fft' methods. The linear convolution methods tend to underestimate the boundaries, as they correspond to zero-padding the image berfore applying a certain Gaussian filter. 'fft' tends to give periodic boundary effects.

The frequency methods apply an n-dimensional transform and so can be used for continous signals with any number of dimensions. The matrix methods are 2D only.

## Getting Started with the Standardizing Transform
To get started, IMED.ST_all contains a wrapper function standardizingTrans(imgs,sigma,method,eps=0,inverse=False)
Here is the doc:

    Takes sequence of images imgs and returns the Spatial Standardized Transform of all images.
    Methods 'full' and 'sep' are 2D methods.
    
    Parameters:
    * imgs (3D array) is a sequence of images to be transformed with dimensions (T,M,N)
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

    with fft.set_workers(-1):
        standardizingTrans(imgs,sigma,method,eps=0,inverse=False)
        
When the number of workers is set to a negative integer (like above), the number of workers is set to os.cpu_count().

Scipy also supports computations using another backend, like pyFFTW:
    
     import pyfftw
     with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        imgs_ST = standardizingTrans(imgs,sigma=(10.,20.),method='DCT',eps=0,inverse=False)

        
