#!/usr/bin/env python3 
import numpy as np
from paddings import pad_values, pad_wrap, pad_edge, pad_reflect, pad_fft
from utils import shift_center
from DFT import DFT2, IDFT2


def conv2(img, kernel, pad):
    """
    Perform two-dimensional convolution/cross-correlation.
    If input is RGB image, result will be converted to signed 16-bit int.
    Convolution is performed on each RGB color channel separately and the results
    are then combined to create resulted RGB image.
    Filter kernel is not flipped for convolution.
    Treats convolution as cross-correlation.

    Inputs:
    - img: ndarray of input image. Can be grayscale or RGB image
    - kernel: 2D kernel used as filter
    - pad: str; padding types to be used for convolution.
    four padding types:
      - clip: pads with 0s
      - wrap: pads with the wrap of the image. treat image as periodic
      - edge: pads the image using image's edges
      - reflect: pads with the reflection of image along the edges
    
    Output:
    Same size (as original) convoluted image
    """
    if np.ndim(kernel) != 2:
        raise ValueError("filter kernel must have 2 dim. %s has %d" % \
                             (kernel, np.ndim(kernel)))
    if np.ndim(img) == 3:
        res = []
        for channel in range(img.shape[2]):
            # convert each resulted layer to int value
            # allow negative values to allow edge detection filters
            layer = conv2(img[:,:,channel], kernel, pad)
            res.append(layer)
        return np.dstack((res[0], res[1], res[2]))

    elif np.ndim(img) == 2:
        H, W = img.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        k_flat = kernel.flatten()
        res = np.zeros((H, W), dtype=np.float32)

        if pad == "clip":
            padded = pad_values(img, (pad_h, pad_w), 0)
        elif pad == "wrap":
            padded = pad_wrap(img, (pad_h, pad_w))
        elif pad == "edge":
            padded = pad_edge(img, (pad_h, pad_w))
        elif pad == "reflect":
            padded = pad_reflect(img, (pad_h, pad_w))
        else:
            raise ValueError("%s is not a valid padding type" % (pad))

        for idx in range(H * W):
            i = idx // W
            j = idx % W
            res[i,j] = np.sum(padded[i:i+k_h, j:j+k_w].flatten() * k_flat)
        return res
    else:
        raise ValueError("Can only take 2D or 3D images as input")


def lin_conv(img, kernel):
    """
    Perform linear convolution in frequency domain
    Procedure:
    - for 2D image; for 3D image, same procedure repeated
    for each layer then combined the layers
    1. pad original iamge to have size PxQ (P = 2*H, Q = 2*W)
       - paddings are reflections of original image over the edges
       - original image is at top left of padded image
    2. pad kernel to have size PxQ
       - use 0 paddings for kernel
       - kernel is at center of the padded kernel
    3. transform padded image to frequency domain
    4. transform padded kernel to frequency domain
    5. perform linear convolution in frequency domain
    6. transform convolution result to spatial domain
    7. shift the transformed result to have top left (0,0) origin
    8. filtered result is on the left corner

    Inputs:
    - img: original 2D or 3D image array
    - kernel: 2D kernel used for filtering
    Output:
    - Convoluted image with same size as the original
    """
    H, W = img.shape[:2]
    kh, kw = kernel.shape

    if np.ndim(kernel) != 2:
        raise ValueError("filter kernel must have 2 dim. %s has %d" % \
                             (kernel, np.ndim(kernel)))
    if np.ndim(img) == 3:
        res = []
        for i in range(3):
            layer = lin_conv(img[:,:,i], kernel)
            res.append(layer)
        return np.dstack((res[0], res[1], res[2]))

    if np.ndim(img) == 2:
        padded = pad_fft(img, (H*2, W*2))
        pkernel = np.zeros(padded.shape)
        # align kernel to be in the center of padded kernel
        ph, pw = padded.shape
        pmy, pmx = ph//2, pw//2
        kmy, kmx = kh//2, kw//2
        min_my = min(pmy, kmy)
        min_mx = min(pmx, kmx)
        min_h, min_w = min(kh, ph), min(kw, pw)
        tmp = kernel[kmy-min_my:kmy-min_my+min_h, kmx-min_mx:kmx-min_mx+min_w]
        pkernel[pmy-min_my:pmy-min_my+min_h, pmx-min_mx:pmx-min_mx+min_w] = tmp

        imgF = DFT2(padded)
        kernelF = DFT2(pkernel)
        cnv = imgF * kernelF
        result = IDFT2(cnv)
        result = shift_center(result)
        return result[:H, :W]
            
