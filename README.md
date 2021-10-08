# IMage Euclidean Distance (IMED)

## Introduction
The IMED is the Euclidean distance (ED) applied to a transformed version of an image or n-dimensional volume that has image-like correlation along axes. It solves some of the shortcommings of using the  pixel-wise Euclidean distance in classification or regression problems. Small displacements do not have as large an impact on the similarity measure when IMED is used over the ED. 

Below, two binary 2D images are displayed, each in two versions. One image has white pixels that are _slightly displaced_ compared to the other.
With the original 'sharper' images, the ED between these two images is large, since ED does not take into acocunt any surroundings of each pixel. This is despite the obvious similarity to the naked eye. The 'blurred' versions are standardizing transformed (ST) with a certain Gaussian filter. The displacement is not penalized as harshly by the ED on these images. 

**This is the IMED: The Euclidean distance on ST images.**
<p align="center">
<img src="https://raw.githubusercontent.com/jfelding/IMED/assets/readme_assets/L2_images/L2_imgs.png" alt="Forward transform of 2D images alter the L2 distance, reduces noise" width="500px" style="horisontal-align:middle">
</p>
## The IMED package
This package contains efficient python implementations of the IMED (distance, transforms) and adds robust inverse transforms for first-ever utility in regression problems e.g. spatio-temporal forecasting.

`IMED.legacy` contains legacy transforms, i.e. slower or less useful versions of the IMED, for historical reasons.

Implementations are based on and extend the work of [On the translation‑invariance of image 
distance metric](https://link.springer.com/content/pdf/10.1186/s40535-015-0014-6.pdf) (Bing Sun, Jufu Feng and Guoping Wang, 2015, Springer).

The ST is a convolution-based transformation. This package therefore implements the transforms in frequency space. Fourier Transform (FFT) and Discrete Cosine Transform (DCT) methods are available, with slightly different edge effects as a result. The DCT is recommended for natural images since it performs _symmetric convolution_. The frequency-based methods allow parallelization and distributed computations if needed.

In the future, an even more efficient finite-support version of the transforms will be added (see [this issue](https://github.com/jfelding/IMED/issues/1))

## Use Cases

### Classification

### Regression Problems

### _n_-Dimensional Problems

## Forward Standardizing Transform

## Backward Standardizing Transform

## Distance Calculation

## Tunable Parameters

## Getting Started with the Standardizing Transform
**Installation**:
Install the latest release from pypi:

    pip install IMED

## Parallelization of frequency methods and backend change
The IMED methods are implemented using the `scipy.fft` module.
By chopping up the transforms into smaller chunks, SciPy supports parallelization by specifying the _workers_ environment variable:
    
    from scipy import fft
    with fft.set_workers(-1):
        standardizingTrans(imgs,sigma,method,eps=0,inverse=False)
        
When the number of workers is set to a negative integer (like above), the number of workers is set to os.cpu_count().

SciPy also supports computations using another backend. For example, we can use [pyFFTW](http://pyfftw.readthedocs.io/en/latest/) as backend like so: 
     
     from scipy import fft
     import pyfftw
     with fft.set_backend(pyfftw.interfaces.scipy_fft):
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        imgs_ST = standardizingTrans(imgs,sigma=(10.,20.),method='DCT',eps=0,inverse=False)
        
## References
[Bing Sun, Jufu Feng, and Guoping Wang. “On the Translation-Invariance of Image
Distance Metric”. In: Applied Informatics 2.1 (Nov. 25, 2015), p. 11.
0089.
ISSN :
2196-
DOI : 10.1186/s40535- 015- 0014- 6 . URL : https://doi.org/10.1186/
s40535-015-0014-6](https://link.springer.com/content/pdf/10.1186/s40535-015-0014-6.pdf)
