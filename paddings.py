#!/usr/bin/env python3
import numpy as np
from utils import get_pad_size


def pad_values(img, size, pad_values):
    """
    Pads image with a constant
    Mimicks the behavior of np.pad(array, size, 'constant')

    Inputs:
    - img: 2D 2D array of input image need to be padded
    - size: int, list or tuple; the size of padding
    - pad_values: int or float; value used for padding

    Output: padded image of size (H + 2 * pad_height, W + 2 * pad_width)
    """
    H, W = img.shape
    pad_h, pad_w = get_pad_size(size)
    res = np.full((H + pad_h*2, W + pad_w*2), pad_values, dtype=np.float32)
    res[pad_h:pad_h+H, pad_w:pad_w+W] = img
    return res


def pad_wrap(img, size):
    """
    Treats image as if it is periodic.
    Mimicks the behavior of np.pad(array, size, 'wrap').
    Allow padding size that is bigger than the input image size

    Inputs:
    - img: 2D array of input need to be padded 
    - size: int, list or tuple; the size of padding
    Output: padded image of size (H + 2 * pad height, W + 2 * pad width)
    """
    H, W = img.shape
    pad_h, pad_w = get_pad_size(size)

    res = img.copy()
    while res.shape[1] < 2 * pad_w + W:
        res = np.hstack((img, res, img))

    tmp = res.copy()
    while res.shape[0] < 2 * pad_h + H:
        res = np.vstack((tmp, res, tmp))

    h, w = res.shape
    start_r = (h - 2 * pad_h - H) // 2
    start_c = (w - 2 * pad_w - W) // 2
    end_r = start_r + 2 * pad_h + H
    end_c = start_c + 2 * pad_w + W
    return res[start_r:end_r, start_c:end_c]


def pad_edge(img, size):
    """
    Pads using the edge values of the image. 
    Mimicks the behavior of np.pad(array, size, 'edge').
    
    Inputs:
    - img: 2D array of input image
    - size: int, list or tuple; the size of padding
    Output: padded image of size (H + 2 * pad height, W + 2 * pad width)
    """
    H, W = img.shape
    pad_h, pad_w = get_pad_size(size)
    res = np.zeros((H + 2 * pad_h, W + 2 * pad_w))

    res[pad_h:pad_h+H, pad_w:pad_w+W] = img   # fill in image
    # pad sizde values
    res[0:pad_h, pad_w:pad_w+W] = img[0,:]    # pad top
    res[pad_h+H:, pad_w:pad_w+W] = img[-1,:]  # pad bottom
    res[pad_h:pad_h+H, 0:pad_w] = img[:,0].reshape((H,1))    # pad left
    res[pad_h:pad_h+H, pad_w+W:] = img[:,-1].reshape((H,1))  # pad right
    # fill in corner values
    res[0:pad_h, 0:pad_w] = img[0,0]          # top left corner
    res[0:pad_h, pad_w+W:] = img[0,-1]        # top right corner
    res[pad_h+H:, 0:pad_w] = img[-1,0]        # bottom left corner
    res[pad_h+H:, pad_w+W:] = img[-1,-1]      # bottom right corner

    return res


def pad_reflect(img, size):
    """
    Pads using the reflection of the image along its edges.
    Mimicks the behavior of np.pad(array, size, 'symmetric')
    Allow padding of size bigger than the size of input image

    Inputs:
    - img: 2D array of input image
    - size: int, list or tuple; the size of padding
    Output: padded image of size (H + 2 * pad height, W + 2 * pad width)
    """
    H, W = img.shape
    pad_h, pad_w = get_pad_size(size)

    res = img.copy()
    tmp = res.copy()  # hold copy of img to allow flipping periodically
    while res.shape[1] < 2 * pad_w + W:
        tmp = np.flip(tmp, axis=1)  # flip horizontally
        res = np.hstack((tmp, res, tmp))

    tmp = res.copy()  # hold copy of res to allow flipping periodically
    while res.shape[0] < 2 * pad_h + H:
        tmp = np.flip(tmp, axis=0)  # flip vertically
        res = np.vstack((tmp, res, tmp))

    h, w = res.shape
    start_r = (h - 2 * pad_h - H) // 2
    start_c = (w - 2 * pad_w - W) // 2
    end_r = start_r + 2 * pad_h + H
    end_c = start_c + 2 * pad_w + W
    return res[start_r:end_r, start_c:end_c]


def pad_fft(img, size):
    """
    padding function specifically for padding image
    before DFT to avoid wrap around error when doing 
    filtering in frequency domain
    Padding will always be mirror images over edges
    Similar behavior to pad_reflect but the original image
      is on top left of the padded result
    Padded.shape = 2 * img.shape
    
    Input: 
    - img: 2D image array
    - size: size of padded image 
    Output:
    - padded image of input with origin = (0,0)
    """
    H, W = img.shape
    nH, nW = get_pad_size(size)
    res = img.copy()
    tmp = img.copy()
    while res.shape[1] < nW:
        tmp = np.flip(tmp, axis=1)
        res = np.hstack((res,tmp))

    tmp = res.copy()
    while res.shape[0] < nH:
        tmp = np.flip(tmp, axis=0)
        res = np.vstack((res, tmp))

    return res[:nH, :nW]
