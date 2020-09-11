#!/usr/bin/env python3
import numpy as np

def DFT2(image):
    """
    Implementation of 2D FFT using a numpy built-in FFT
    Input:
    - image: 2-D image of grayscale image that need to be transformed

    Output:
    - 2-D matrix of fourier transform of original image
    """
    # only take grayscale image or 2D inputs
    assert(np.ndim(image) == 2)

    M, N = image.shape
    res = np.zeros((M,N), dtype=np.complex64)
    # transform rows then columns
    for i in range(M):
        res[i,:] = np.fft.fft(image[i,:])
    for i in range(N):
        res[:,i] = np.fft.fft(res[:,i])
    return res


def IDFT2(F):
    """
    Implementation of 2D IDFT using the above 2D DFT function
    This is done by inputing Fourier Transform into DFT and divide 
    the result by M*N
    
    Input:
    - F: 2D signals in frequency domain
    Output:
    - 2D image of the image in spatial domain
    """
    # only take 2-D inputs
    assert(np.ndim(F) == 2)

    M, N = F.shape
    res = DFT2(F) / (M * N)
    res = res.astype(np.float)
    return np.flip(res)
