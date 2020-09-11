#!/usr/bin/env python3 
import numpy as np

def get_pad_size(size):
    """
    Unpack the padding size
    Input:
    - size: int or list or tuple
    Output:
    - tuple of pad height and pad width
    """
    if isinstance(size, (list, tuple)):
        if len(size) != 2:
            raise ValueError("size needs 2 values to unpack. %s has %d" % \
                                 ((size), len(size)))
        pad_h, pad_w = size
    elif isinstance(size, int):
        pad_h = pad_w = size
    else:
        raise ValueError("%s is not a valid size type" % (size))

    return (pad_h, pad_w)


def pad_values(img, size, pad_values):
    """
    Pads image with a constant.
    Mimicks the behavior of np.pad(array, size, 'constant').
    
    Inputs:
    - img: 2D array of input image need to be padded
    - size: int, list or tuple; the size of padding
    - pad_values: int or float; value used for padding

    Output: padded image of size (H + 2 * pad height, W + 2 * pad width)
    """
    H, W = img.shape
    pad_h, pad_w = get_pad_size(size)
    res = np.full((H + pad_h*2, W + pad_w*2), pad_values, dtype=np.float)
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
    res = np.zeros((H + 2*pad_h, W + 2*pad_w))

    res[pad_h:pad_h+H, pad_w:pad_w+W] = img   # fill in image
    # pad side values
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
            layer = conv2(img[:,:,channel], kernel, pad).astype(np.int16)
            res.append(layer)
        return np.dstack((res[0], res[1], res[2]))

    elif np.ndim(img) == 2:
        H, W = img.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        k_flat = kernel.flatten()
        res = np.zeros((H, W))

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
