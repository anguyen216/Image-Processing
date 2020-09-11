#!/usr/bin/env python3
import numpy as np
import matplotlib
# to deal with "python not installed as framework" error
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from skimage.draw import circle_perimeter

def get_pad_size(size):
    """
    Unpack the padding size
    Input:
    - size: int or list or tuple
    Output:
    - tupple of pad height and pad width
    """
    if isinstance(size, (list, tuple)):
        if len(size) != 2:
            raise ValueError("size needs 2 values to unpack. %s has %d" % \
                                 ((size), len(size)))
        pad_h, pad_w = size
    elif isinstance(size, int):
        pad_h = pad_w = size
    else:
        raise ValueError("size needs to be/include int values")

    return (pad_h, pad_w)


def intensity_spread(img, L):
    """
    The intensity transformation function is
    T(x) = (L - 1) * (x - min) / (max - min)
    This is a function written in homework 2
    
    Inputs:
    img: image represented in a 2D array
    L: int; L - 1 is the highest new intensity

    Output:
    an image with intensity transformed to fit in the range [0, L-1]
    """

    pixels = img.flatten()
    min_pix = np.amin(pixels)
    max_pix = np.amax(pixels)
    res = (L - 1) * ((pixels - min_pix) / (max_pix - min_pix))
    return np.reshape(res, img.shape)


def log_shift(data):
    """
    Shift data using transform s = log(1 + abs(d))
    Input:
    - Data: list of data d that needs to be transform
    Output:
    List of transformed data d
    """
    result = [np.log(1 + np.abs(d.copy())) for d in data]
    return result


def exp_shift(data):
    """
    Inverse of the log_shift function
    transform s = exp(d) - 1
    Input:
    - data: list of data d that needs to be transform
    Ouput: list of transformed data d
    """
    result = [np.exp(d) - 1 for d in data]
    return result


def shift_center(img):
    """
    Shift the center origin of an image to top left
    so the image have (0,0) as the origin
    Input:
    - img: 2D image array with origin at (H//2, W//2)
    Output:
    - 2D image array with origin at (0,0)
    """
    H, W = img.shape
    res = img.copy()
    res = np.hstack((res, img))
    res = np.vstack((res, res))
    return res[H//2:H+H//2, W//2:W+W//2]


def visualization(data, rows, cols, titles, figsize):
    """
    Function to visualize the result images.
    This function is a modified version of the code written by 
    user "swatchai" as an answer on stackoverflow. 
    The code and thread can be found at this link:
    https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
    If display color image, make sure color channels is in RGB order

    Inputs:
    - data: list of images array
    - rows: int; number of rows of the plot
    - cols: int; number of columns of the plot
    - titles: list of subplots' titles
    - figsize: int tuples; the size of the entire figure

    Output: a figure contains several images, each in a separate subplot
    """
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    # plot image on each subplot
    for i, axi in enumerate(ax.flat):
        # i is in range [0, nrows * ncols)
        # axi is equivalent to ax[rowid][colid]
        axi.imshow(data[i])
        axi.set_title(titles[i])
    plt.tight_layout(True)
    plt.show()


def gaussian_kernel(sigma):
    """
    - 2D gaussian kernel.
    - mimick the behavior of fspecial('gaussian', [shape], [sigma])
    in MATLAB
    - however, size of the kernel is automatically calculated
    - using kernel_size = 4 * sigma + 1
    - source: https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (by user ali_m)
    """
    size = [int(sigma*4) + 1, int(sigma*4) + 1]  # want odd-sized kernel
    m, n = [(s - 1.) / 2. for s in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0: h /= sumh
    return h


# function to resample original image
def resample(img, newSize):
    """
    Function to downsample and upsample image
    Use simple nearest neighbor interpolation
    Input:
    - img: 2D or 3D image matrix
    - newSize: tuple of result size (height, width)

    Output: resampled image
    """
    # use float as datatype to preserve float precision
    # will convert to np.uint8 when display result
    # get the new dimensions
    nH, nW = newSize
    if np.ndim(img) == 2:
        H, W = img.shape
        res = np.zeros((nH, nW), dtype=np.float32)
    elif np.ndim(img) == 3:
        H, W, _ = img.shape
        res = np.zeros((nH, nW, _), dtype=np.float32)
    else:
        raise ValueError("input image has invalid dimension %s" % (img.shape))

    # interpolate the value for the result
    for idx in range(nH * nW):
        i = idx // nW
        j = idx % nW
        orig_i = int((i * H) // nH)
        orig_j = int((j * W) // nW)
        res[i, j] = img[orig_i, orig_j]
    return res


def circle_area(radius):
    """
    Calculate the area(s) of circle(s)
    Input:
    - radius: int or numpy array of radii
    Output:
    - area of circle(s)
    """
    return np.pi * radius**2


def overlap_area(coords1, coords2):
    """
    Calculate the are overlap of 2 circles
    Inputs:
    - coords1: [y, x, r1, v1] coordinates, radius and pixel value of circle 1
      - np array of all circles 1
    - coords2: [y, x, r2, v2] coordinates, radius and pixel value of circle 2
      - np array of all circles 2
    Output:
    - area of overlap of two circles
    """

    y1 = coords1[:,0]
    x1 = coords1[:,1]
    r1 = coords1[:,2]
    y2 = coords2[:,0]
    x2 = coords2[:,1]
    r2 = coords2[:,2]
    d = np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
    area = np.zeros(coords1.shape[0])

    # one contains the other
    i = d <= np.abs(r1 - r2)
    if np.sum(i) > 0:
        if np.sum(i) > 1:
            area[i] = circle_area(np.minimum(r1[i], r2[i]))
        else:
            area[i] = circle_area(np.min((r1[i], r2[i])))
    # calculating area of intersection between two circles
    # following formula from eq.14 on:
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    i = (np.abs(r1 - r2) < d) & (d < (r1 + r2))
    R = np.maximum(r1[i], r2[i])
    r = np.minimum(r1[i], r2[i])
    a1 = (r**2) * np.arccos((d[i]**2 + r**2 - R**2)/(2*d[i]*r))
    a2 = (R**2) * np.arccos((d[i]**2 + R**2 - r**2)/(2*d[i]*R))
    a3 = 0.5 * np.sqrt((-d[i]+r+R) * (d[i]+r-R) * (d[i]-r+R) * (d[i]+r+R))
    area[i] = a1 + a2 - a3

    # when d == r1 + r2
    # formula from eq.15 on:
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    i = (d == (r1 + r2))
    R = np.maximum(r1[i], r2[i])
    a1 = 2 * (R**2) * np.arccos(d[i]/(2 * R)) 
    a2 = 0.5 * d[i] * np.sqrt(4 * R**2 - d[i]**2)
    area[i] = a1 - a2
    return area
        

def draw_circles(img, circles):
    """
    overlay circles on image
    Inputs:
    - img: 3D image array
    - circles: list of circles to draw; circles have form
      - [y_coord, x_coord, radius, value]
    Output:
    - Overlay circles on image.  Circles borders are in red
    """
    red = (1.0, 0.0, 0.0)
    H, W = img.shape[:2]

    for c in circles:
        y, x = int(c[0]), int(c[1])
        r = int(np.ceil(c[2]))
        cv2.circle(img, (x,y), r, red, 2)
