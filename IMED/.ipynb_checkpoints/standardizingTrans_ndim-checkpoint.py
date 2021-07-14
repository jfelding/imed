import numpy as np
from scipy.fft import dct, idct
#from scipy.fft import rfftn, irfftn
from scipy.fft import rfftn, irfftn, fftfreq,rfftfreq


NA = np.newaxis

def axis_split(A,axis):
    (Am,Ai,An) = (int(np.prod(A.shape[:axis])),  A.shape[axis], int(np.prod(A.shape[axis+1:])));
    return (Am,Ai,An)

def ST_1dim_DCT(Img,sigma,d=0,eps=0,inverse=False,eps_norm=True):
    # Gaussian in k-space
    k_d     = np.linspace(0,np.pi,Img.shape[d],dtype=Img.dtype)
    if eps_norm:
        g12_dct = (np.exp(- k_d**2*sigma**2/4) + eps) / (1 + eps)
    else:
        g12_dct = np.exp(- k_d**2*sigma**2/4) + eps

    # Transform to k-space
    img_dct = dct(Img, axis=d,type=1).reshape( axis_split(Img,d) )
    
    if inverse:
        Img_folded_k = img_dct / g12_dct[NA,:,NA]
    else:
        Img_folded_k = img_dct * g12_dct[NA,:,NA]

    Img_folded = idct(Img_folded_k,axis=1,type=1)

    return Img_folded.reshape(Img.shape)
    
def ST_1dim_FFT(Img,sigma,d=0,eps=0,eps_norm=True,inverse=False):   
    # Transform to k-space
    #img_fft = rfftn(Img)
   
    img_fft = rfftn(Img)

    fft_shape = img_fft.shape
    img_fft = img_fft.reshape(axis_split(img_fft,d))
    
    # if last axis, need other k definition for rfft
    if d == Img.ndim-1:
        k_d = rfftfreq(2*fft_shape[d]-1)*2*np.pi
        # ensure correct dtype
        k_d = np.asarray(k_d,dtype=Img.dtype)

    else:
        k_d = fftfreq(fft_shape[d])*2*np.pi  
        # ensure correct dtype
        k_d = np.asarray(k_d,dtype=Img.dtype)

            
    # Gaussian in k-space
    if eps_norm:
        g12_fft = (np.exp(- k_d**2*sigma**2/4)+eps)/(1+eps)
     
    else:
        g12_fft = np.exp(- k_d**2*sigma**2/4) + eps
    
    if inverse:
        Img_folded_k = img_fft / g12_fft[NA,:,NA]
    else:
        Img_folded_k = img_fft * g12_fft[NA,:,NA]

    Img_folded = irfftn(Img_folded_k.reshape(fft_shape),s=Img.shape)

    return Img_folded.reshape(Img.shape)    
    
def ST_ndim_DCT(imgs,sigma,eps=0.,inverse=False,eps_norm=True):
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
        if eps == 0:
            eps_norm=False
            
        imgs = ST_1dim_DCT(Img=imgs,sigma=sigma[d],d=d,eps=eps,eps_norm=eps_norm,inverse=inverse)
    
    return imgs
        
        
def ST_DCT_by_FFT(imgs, sigma, eps=0.,inverse=False,eps_norm=True):
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
        if eps == 0:
            eps_norm=False
        imgs = ST_1dim_FFT(imgs,sigma=sigma[d],d=d,eps=eps,eps_norm=eps_norm,inverse=inverse)
        
        #Cut to original shape before moving on to other axis         
        imgs = imgs.take(indices=range(orig_shape[d]),axis=d)
    
    return imgs    
 
def ST_ndim_FFT(imgs, sigma, eps=0,eps_norm=True,inverse=False):
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
            
        if eps == 0:
            eps_norm=False        
        
        imgs = ST_1dim_FFT(imgs,sigma=sigma[d],d=d,eps=eps,eps_norm=eps_norm,inverse=inverse)
    
    return imgs
