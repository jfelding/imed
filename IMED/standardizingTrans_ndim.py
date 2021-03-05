import numpy as np
from scipy.fft import dct, idct
#from scipy.fft import rfftn, irfftn
#from jax.numpy.fft import rfftn, irfftn
import numpy as jnp
from scipy.fft import rfftn, irfftn, fftfreq,rfftfreq
from numpy import delete #no jnp version avaliable yet

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

    Img_folded = idct(Img_folded_k,axis=d,type=1)

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
        k_d     = np.linspace(0,np.pi,shape[d])
        g12_dct = np.exp(- k_d**2*sigma[d]**2 / (4)) + eps

        other_dims = tuple(np.delete(all_dims, d))

        g12_dct =  np.expand_dims(g12_dct,axis = other_dims)
        
        # DCT transform along all axes and  
        # repeat OTF for axis-by-axis convolution
        img_dct = dct(imgs, axis=d,type=1)
        for not_d in other_dims:
            g12_dct = np.repeat(g12_dct,repeats=shape[not_d],axis=not_d)
            #img_dct = dct(img_dct, axis=not_d,type=1)
        
        if inverse:
            imgs = idct(img_dct/g12_dct,axis=d,type=1)
       
        else:
            imgs = idct(img_dct*g12_dct,axis=d,type=1)
            
        #for not_d in other_dims:
        #    imgs = idct(imgs,axis=not_d,type=1)
            
    return imgs

def ST_ndim_DCT_by_FFT(imgs, sigma, eps=0.,inverse=False):
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
        imgs_reverse = jnp.flip(imgs, axis=d)
        # for DCT equivalence
        imgs_reverse = imgs_reverse.take(indices=range(1,imgs_reverse.shape[d]-1),axis=d)
        imgs         = jnp.concatenate((imgs,imgs_reverse),axis=d)
        
        img_fft = rfftn(imgs, axes=all_dims)
        
        # last axis will be half of other axes with rfft...
        fft_shape = img_fft.shape
            
        # if last axis, need other k_d for rfft unlike fft
        if d == all_dims[-1]:
            k_d = rfftfreq(2*fft_shape[d]-1)*2*jnp.pi
            if imgs.dtype == jnp.float32:
                k_d = jnp.float32(k_d)
        else:
            # equal to but with same dtype as input
            k_d = fftfreq(fft_shape[d])*2*jnp.pi
            if imgs.dtype == jnp.float32:
                k_d = jnp.float32(k_d)
            
        g12_fft = jnp.exp(- k_d**2*sigma[d]**2 / (4)) + eps

        other_dims = tuple(delete(all_dims, d))

        g12_fft =  jnp.expand_dims(g12_fft, axis = other_dims)

        for not_d in other_dims:
            # probably have to handle fftshift
            g12_fft = jnp.repeat(g12_fft,repeats=fft_shape[not_d],axis=not_d)
            
        if inverse:
            imgs = irfftn(img_fft/g12_fft,axes=all_dims)
        else:
            imgs = irfftn(img_fft*g12_fft,axes=all_dims)
    
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
        
        img_fft = rfftn(imgs, axes=all_dims)
        
        # last axis will be half of other axes with rfft...
        fft_shape = img_fft.shape
        
        """# if last axis, need other k definition for rfft
        if d == all_dims[-1]:
            k_d = jnp.linspace(0,jnp.pi,fft_shape[d],dtype=imgs.dtype)
        else:
            # equal to #jnp.fft.fftfreq(fft_shape[d])*2*jnp.pi but with same dtype as input
            k_d = (jnp.arange(1, fft_shape[d] + 1, dtype=imgs.dtype) // 2) / fft_shape[d] *2*jnp.pi"""
            
        # if last axis, need other k definition for rfft
        if d == all_dims[-1]:
            k_d = rfftfreq(2*fft_shape[d]-1)*2*jnp.pi#jnp.linspace(0,jnp.pi,fft_shape[d],dtype=imgs.dtype)
            if imgs.dtype == jnp.float32:
                k_d = jnp.float32(k_d)
        else:
            # equal to but with same dtype as input
            k_d = fftfreq(fft_shape[d])*2*jnp.pi #(jnp.arange(1, fft_shape[d] + 1, dtype=imgs.dtype) // 2) / fft_shape[d] *2*jnp.pi 
            if imgs.dtype == jnp.float32:
                k_d = jnp.float32(k_d)
                
        g12_fft = jnp.exp(- k_d**2*sigma[d]**2 / (4)) + eps

        other_dims = tuple(delete(all_dims, d))

        g12_fft =  jnp.expand_dims(g12_fft, axis = other_dims)

        for not_d in other_dims:
            
            # probably have to handle fftshift
            g12_fft = jnp.repeat(g12_fft,repeats=fft_shape[not_d],axis=not_d)
            
        if inverse:
            imgs = irfftn(img_fft/g12_fft,axes=all_dims)
        else:
            imgs = irfftn(img_fft*g12_fft,axes=all_dims)
    
        #Cut to original shape before moving on to other axis         
        #imgs = imgs.take(indices=range(orig_shape[d]),axis=d)
    
    return imgs
