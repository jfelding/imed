from IMED.standardizingTrans_ndim import ST_ndim_DCT, ST_ndim_FFT, ST_ndim_DCT_by_FFT
from IMED.spatial_ST import ST_fullMat, ST_sepMat
def standardizingTrans(imgs,sigma,method='dct',eps=0,inverse=False):
    """
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
    
    if  method == 'full':
        if inverse==True:
            print("No inverse method implemented")
            return
        imgs_ST = ST_fullMat(imgs,sigma,eps)
    
    elif method == 'sep':
        if inverse==True:
            print("No inverse method implemented")
            return
        imgs_ST = ST_sepMat(imgs,sigma,eps)

    elif method == 'fft':
        imgs_ST = ST_ndim_FFT(imgs, sigma, eps,inverse)
        
    elif method == 'dct_by_fft':
        imgs_ST = ST_ndim_DCT_by_FFT(imgs, sigma, eps,inverse)
    
    elif method == 'dct':
        imgs_ST = ST_ndim_DCT(imgs, sigma, eps, inverse)
        
    else:
        print(f'Invalid method "{method}". Choosing dct.')
        method = 'dct'
        standardizingTrans(imgs,sigma,method,eps,inverse)    
    
    return imgs_ST
