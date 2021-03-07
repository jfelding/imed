import numpy as np
from scipy.fft import dct, idct
#from scipy.fft import rfftn, irfftn
#from jax.numpy.fft import rfftn, irfftn
import jax.numpy as jnp
from scipy.fft import rfftn, irfftn, fftfreq,rfftfreq


NA = np.newaxis

def axis_split(A,axis):
    (Am,Ai,An) = (int(np.prod(A.shape[:axis])),  A.shape[axis], int(np.prod(A.shape[axis+1:])));
    return (Am,Ai,An)

def ST_1dim_DCT(Img,sigma,d=0,eps=0,inverse=False):
    # Gaussian in k-space
    k_d     = np.linspace(0,np.pi,Img.shape[d])
    g12_dct = np.exp(- k_d**2*sigma**2/4) + eps

    # Transform to k-space
    img_dct = dct(Img, axis=d,type=1).reshape( axis_split(Img,d) )
    
    if inverse:
        Img_folded_k = img_dct / g12_dct[NA,:,NA]
    else:
        Img_folded_k = img_dct * g12_dct[NA,:,NA]

    Img_folded = idct(Img_folded_k,axis=1,type=1)

    return Img_folded.reshape(Img.shape)
    
def ST_1dim_FFT(Img,sigma,d=0,eps=0,inverse=False,jax_backend=False):   
    # Transform to k-space
    #img_fft = rfftn(Img)
    if jax_backend:
        img_fft = jnp.fft.rfftn(Img)
    else:   
        img_fft = rfftn(Img)

    fft_shape = img_fft.shape
    img_fft = img_fft.reshape(axis_split(img_fft,d))
    
    # if last axis, need other k definition for rfft
    if d == Img.ndim-1:
        k_d = rfftfreq(2*fft_shape[d]-1)*2*jnp.pi
        if Img.dtype == jnp.float32:
            k_d = jnp.float32(k_d)
    else:
        k_d = fftfreq(fft_shape[d])*2*jnp.pi  
        if Img.dtype == jnp.float32:
            k_d = jnp.float32(k_d)
            
    # Gaussian in k-space
    g12_fft = np.exp(- k_d**2*sigma**2/4) + eps
    
    if inverse:
        Img_folded_k = img_fft / g12_fft[NA,:,NA]
    else:
        Img_folded_k = img_fft * g12_fft[NA,:,NA]
    print()
    if jax_backend:
        Img_folded = jnp.fft.irfftn(Img_folded_k.reshape(fft_shape))
    else:
        Img_folded = irfftn(Img_folded_k.reshape(fft_shape))

    return Img_folded.reshape(Img.shape)    
    
def ST_ndim_DCT(imgs,sigma,eps=0.,inverse=False):
    # automatic d-dimensional standardizing transform
    # via DCT, i.e. symmetric boundary conditions
    # eps is an optional constant added to the OTF to reduce  
    # noise amplification when deconvolving
    shape = imgs.shape
    dims     = len(shape)
    all_dims = range(dims)
    
    
    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        #if sigma is a scalar, it will be used for all axes
        sigma = np.ones(dims)*sigma
        
    # do convolution, axis by axis
    for d in range(dims):
        if sigma[d]==0:
            #convolution has no effect
            continue

        if shape[d]<2:
            #cant do convolution along this axis 
            continue
        
        imgs = ST_1dim_DCT(Img=imgs,sigma=sigma[d],d=d,eps=eps,inverse=inverse)
        
        
def ST_DCT_by_FFT(imgs, sigma, eps=0.,inverse=False,jax_backend=True):
    # automatic d-dimensional standardizing transform
    # via FFT. Uses per-axis mirroring to reduce edge discontinuities
    # eps is an optional constant added to the OTF to reduce  
    # noise amplification when deconvolving

    orig_shape    = imgs.shape
    dims     = len(orig_shape)
    all_dims = range(dims)

    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        #if sigma is a scalar, it will be used for all axes
        sigma = np.ones(dims)*sigma
    
    for d in range(dims):
        if sigma[d]==0:
            #convolution has no effect
            continue        
        if orig_shape[d]<3:
            print(f'Skipping dimension d={d} with size {orig_shape[d]}')
            #cant do convolution along this axis 
            continue
        
        #mirror image along axis d
        imgs_reverse = np.flip(imgs, axis=d)
        # for DCT equivalence
        imgs_reverse = imgs_reverse.take(indices=range(1,imgs_reverse.shape[d]-1),axis=d)
        imgs         = np.concatenate((imgs,imgs_reverse),axis=d)
        
        imgs = ST_1dim_FFT(imgs,sigma[d],d,eps,inverse,jax_backend)
        
        #Cut to original shape before moving on to other axis         
        imgs = imgs.take(indices=range(orig_shape[d]),axis=d)
    
    return imgs    
 
def ST_ndim_FFT(imgs, sigma, eps=0.,inverse=False):
    # automatic d-dimensional standardizing transform
    # via FFT. Uses per-axis mirroring to reduce edge discontinuities
    # eps is an optional constant added to the OTF to reduce  
    # noise amplification when deconvolving
    orig_shape    = imgs.shape
    dims     = len(orig_shape)
    all_dims = range(dims)
    
    # Make sigma d-dimensional if not already
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, they must match number of axes
        assert len(sigma) == dims
    else:
        #if sigma is a scalar, it will be used for all axes
        sigma = np.ones(dims)*sigma
    
    for d in range(dims):
        if sigma[d]==0:
            #convolution has no effect
            continue
            
        if orig_shape[d]<2:
            #cant do convolution along this axis 
            continue
        
        imgs = ST_1dim_FFT(imgs,sigma[d],d,eps,inverse,jax_backend)
    
    return imgs
