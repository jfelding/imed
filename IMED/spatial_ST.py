import numpy as np

def ST_fullMat(imgs,sigma,inverse=False):
    # This is created by Niklas Heim / James Avery, and bugfixed by Jacob Felding
    # This method is present for historical reasons. ST_sepMat() is much faster,
    # and should be used instead.
    
    # Purposefully not M, N

    #sigma check
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # if multiple sigmas are given, there should be 2 for spatial transform
        print('This method allows only single sigma values. Using the first one')
        sigma = sigma[0]
    if len(imgs.shape) == 3:
        N, M = imgs.shape[1:]
    else:
        N, M = imgs.shape
        imgs = np.expand_dims(imgs,axis=0)
    
    X = np.arange(M,dtype=imgs.dtype)
    Y = np.arange(N,dtype=imgs.dtype)

    P = (X[None, :, None, None] - X[None, None, None, :])**2 \
        + (Y[:, None, None, None] - Y[None, None, :, None])**2

    G = 1 / (2 * np.pi * sigma**2) * np.exp(- P / (2 * sigma**2)) 
    
    #Construct NM x NM G matrix
    G = G.reshape((M * N, M * N))
    
    # Use eigenvalue decomposition to construct G^1/2 used for standardizing transform
    (w,V) = np.linalg.eigh(G)
    # G^(1/2) has eigenvalues that are the square root of eigenvalues of G
    w = np.maximum(np.zeros(len(w),dtype=imgs.dtype),w)
    s = np.sqrt(w)
    
    # Eigenvalue decomposition
    #G_sqrt  = np.dot(V, s[:,None]*V.T)
    
    if inverse:
        VTS = -s[:,None]*V.T
    else:
        VTS = s[:,None]*V.T
    
    return np.array([np.dot(V, np.dot(VTS,z.reshape(-1))) for z in imgs]).reshape(-1,N,M)



def ST_sepMat(imgs,sigma,inverse=False):
    '''Implements standardizing transform of image sequence using kronecker product matrix separation'''
    
    # Make sigma 2-dimensional if not already
    if isinstance(sigma, (list, tuple, np.ndarray)):
        # Should be 2 sigmas for spatial transform
        assert len(sigma) == 2
    else:
        #if sigma is a scalar, it will be used for all axes
        sigma = np.ones(2)*sigma
    
    
    if len(imgs.shape) == 3:
        N, M = imgs.shape[1:]
    else:
        N, M = imgs.shape

    # Create 1D arrays corresponding to image dimensions
    X = np.arange(M,dtype=imgs.dtype)
    Y = np.arange(N,dtype=imgs.dtype)  

    # Create 2D arrays where P_x(i,j) denotes (x_i-x_j)**2
    P_x = (X[None,:] - X[:,None])**2 
    P_y = (Y[None,:] - Y[:, None])**2
    
    # Create G decompositions such that G = np.kron(G_x,G_y)
    # This way we store NxN, MxM matrices, not NMxNM
    G_x =  1 / (np.sqrt((2 * np.pi ))*sigma[0]) * np.exp(- P_x / (2 * sigma[0]**2))
    G_y =  1 / (np.sqrt((2 * np.pi ))*sigma[0]) * np.exp(- P_y / (2 * sigma[1]**2)) 
    
    # Determine eigenvectors and of G using that G = np.kron(G_x,G_y)
    # The matrices are symmetric since (x_i-x_j)**2 = (x_j-x_i)**2
    eigvals_Gx,eigvecs_Gx = np.linalg.eigh(G_x)
    eigvals_Gy,eigvecs_Gy = np.linalg.eigh(G_y)
    
    # For robustness to negative eigenvalues that should not arise as
    #G is positive definte, but may for numerical reasons
    eigvals_Gx = np.maximum(eigvals_Gx,np.zeros(M,dtype=imgs.dtype))
    eigvals_Gy = np.maximum(eigvals_Gy,np.zeros(N,dtype=imgs.dtype))

    # For the Standardizing Transfrom G^(1/2)x (which blurs image x) 
    # we need matrix G^(1/2) Such that G = G^(1/2)G^(1/2)
    # Again decomposition allows G^(1/2) = np.kron(G_sqrt_x,G_sqrt_y)
    # Below, use eigendecomposition of G to construct G_sqrt_x, G_sqrt_y

       
    if inverse:
        G_sqrt_x = np.dot(eigvecs_Gx, -np.sqrt(eigvals_Gx)[:,None]*eigvecs_Gx.T) 
        G_sqrt_y = np.dot(eigvecs_Gy, -np.sqrt(eigvals_Gy)[:,None]*eigvecs_Gy.T)
    else:
        G_sqrt_x = np.dot(eigvecs_Gx, np.sqrt(eigvals_Gx)[:,None]*eigvecs_Gx.T) 
        G_sqrt_y = np.dot(eigvecs_Gy, np.sqrt(eigvals_Gy)[:,None]*eigvecs_Gy.T) 

    return [np.dot(G_sqrt_x,np.dot(z,G_sqrt_y)) for z in imgs]
