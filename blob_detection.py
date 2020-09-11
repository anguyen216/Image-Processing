#!/usr/bin/env python3
import numpy as np
import convolution as conv
import utils
from scipy import ndimage
from itertools import combinations
from skimage import io, color
import argparse
import time


def project_points(original, scale, points):
    """
    Given (y,x) coordinates of a point, project it onto 
    image of different scale

    Inputs:
    - original: image array of image for points to be projected onto
    - scale: the scale between image that contains points and original image
    - points: list of coordinates in form [y, x, radius, signal_values]

    Output:
    - new points with (y, x) and radius scale to fit onto the new image
    """
    projected = []
    H, W = original.shape[:2]
    h, w = H//scale, W//scale
    for coord in points:
        y, x, rad, v = coord
        orig_y = int((y * H) // h)
        orig_x = int((x * W) // w)
        projected.append([orig_y, orig_x, rad*scale, v])
    return projected


def layer_peaks(octave):
    """
    Given an octave, find signals extrema within layer and across layers

    Input:
    - octave: 3D numpy array of shape (image_height, image_width, num_layers)

    Output:
    - signals extrema in each layer
    """
    n = octave.shape[2]
    # true_peaks - peaks of layer after compare to 
    # 18 neighboring pixels in below and above layers
    # peaks - peaks of layer after comparing to 8 neighboring pixels 
    true_peaks = np.zeros(octave.shape)
    peaks = np.zeros(octave.shape)

    # find peaks within layer
    # normalize for easy thresholding
    for i in range(n):
        tmp = utils.intensity_spread(octave[:,:,i], 2)
        mask = ndimage.maximum_filter(tmp, 3)
        mask[mask <= 0.25] = 0  # mask out weak signals
        mask[mask == tmp] = 1
        mask[mask != 1] = 0
        peaks[:,:,i] = mask * octave[:,:,i]
    
    # find peaks across layers
    for i in range(n):
        current = peaks[:,:,i].copy()
        if i == 0:
            top = peaks[:,:,i+1]
            current[current <= top] = 0
        elif i == n-1:
            bottom = peaks[:,:,i-1]
            current[current <= bottom] = 0
        else:
            bottom = peaks[:,:,i-1]
            top = peaks[:,:,i+1]
            current[current <= bottom] = 0
            current[current <= top] = 0
        current[current != 0] = 1
        true_peaks[:,:,i] = current * peaks[:,:,i]
    return true_peaks


def build_octave(img, init_sigma, k, num_layers):
    """
    Construct a single octave for scale space

    Inputs:
    - image: 2D array of base image used to build octave
    - init_sigma: initial sigma value before applying scaling factor
    - k: scaling factor
    - num_layers: int; number of layer in a DoG octave

    Output:
    - octave: 3D array of shape (img_height, img_width, num_layers)
      - this octave of DoG
    - sigmas: corresponding sigma used in each layer
    """
    H, W = img.shape[:2]
    octave = np.zeros((H, W, num_layers))
    G = []
    for i in range(num_layers + 1):
        kernel = utils.gaussian_kernel(init_sigma * (k**i))
        G.append(conv.lin_conv(img, kernel))
        if i > 0:
            octave[:,:,i-1] = (G[i] - G[i-1])**2
    sigmas = [init_sigma * (k**i) for i in range(num_layers)]
    return octave, sigmas


def build_scale_space(img, init_sigma, k, num_layers, num_oct):
    """
    Construct Laplacian scale space
    - kernel used for smoothing before downsampling is fixed with sigma = 2
    - always downsample by downscale = 1/2
    - base sigma of each octave = 2 * sigma of previous octave

    Inputs:
    - img: 2D array of original image
    - init_sigma: initial sigma value before applying scaling factor
                  and downsampling between octave
    - k: scaling factor
    - num_layers: int; number of layers in each octave
    - num_oct: int; number of octave in the scale space

    Output:
    - space: Laplacian scale space where
             len(space) == num_oct
             space[i].shape = (img_height, img_width, num_layers)
    - sigmas: corresponding sigmas for layers and octaves
    """
    sigmas = []
    space = []
    base = img.copy()
    smoothing = utils.gaussian_kernel(2)
    for i in range(num_oct):
        sigma = init_sigma * (2**i)
        octave, s = build_octave(base, sigma, k, num_layers)
        h, w = base.shape[:2]
        base = utils.resample(conv.lin_conv(base, smoothing), (h//2, w//2))
        space.append(octave)
        sigmas.append(s)
    return space, sigmas


def find_peaks(space, sigmas, threshold):
    """
    Find peaks/blobs center within octave in space
    
    Inputs:
    - space: list of np.array; Laplacian scale space
    - sigmas: list; sigmas of octaves and octaves layers in scale space
    - threshold: threshold of intersection-over-union area
                 used to suppress significantly overlaping blobs

    Outputs:
    - peaks: list of dictionary; peaks have form [y, x, radius, signal_value]
             each dictionary represent peaks of an octave
    """
    # radius = sigma * np.sqrt(2)
    n = len(space)  # number of octaves
    peaks = []
    for i in range(n):
        s = sigmas[i]
        oct = layer_peaks(space[i])
        tmp_peaks = dict()
        for j in range(oct.shape[2]):
            layer = oct[:,:,j]
            coords = np.argwhere(layer > 0).tolist()
            if coords != []:
                # add radius and loc value to blob locations
                m = np.sqrt(2)
                tmp = [c+[np.ceil(m*s[j]), layer[c[0],c[1]]] for c in coords] 
                nonmax_suppress(tmp_peaks, tmp, threshold)
        oct_peaks = dict()
        tmp_peaks = list(tmp_peaks.values())
        nonmax_suppress(oct_peaks, tmp_peaks, threshold)
        peaks.append(oct_peaks)
    return peaks


def update_peaks(peaks_dict, peaks, rejects):
    """
    Given dictionary of true extrema, update dictionary using 
    new found peaks and rejecting peaks

    Inputs:
    - peaks_dict: dictionary of true_extrema
    - peaks: list of list; list of information of new found peaks
    - rejects: list of list; list of information of extrema to be deleted
               from dictionary

    Outputs:
    - udpated peaks dictionary
    """
    n = len(peaks)
    for i in range(n):
        circle = peaks[i]
        key = str(circle[:2])
        peaks_dict[key] = circle
    for circle in rejects:
        key = str(circle[:2])
        if key in peaks_dict:
            del peaks_dict[key]


def nonmax_suppress(true_peaks, extremas, threshold):
    """
    Given current peaks dictionary, potential extrema and threshold used
    for suppression, suppress false extrema and update peaks dictionary
    
    Inputs:
    - true_peaks: current dictionary of true extrema to be updated
    - extremas: list of potential extrema
    - threshold: value used to suppress false extrema
    
    Outputs:
    - updated true_peaks dictionary with false extrema removed and 
      true peaks added
    """
    # threshold is for intersection-over-union
    # sort potential extrema by their radii in increasing order
    idx = np.argsort(np.array(extremas)[:,2])
    extremas = np.array(extremas)[idx].tolist()
    comb = list(combinations(extremas, 2))
    if len(comb) == 0:
        return

    circles1 = np.array([c[0] for c in comb])
    circles2 = np.array([c[1] for c in comb])
    area1 = utils.circle_area(circles1[:,3])
    area2 = utils.circle_area(circles2[:,3])
    area = utils.overlap_area(circles1, circles2)
    
    # if overlap is less than threshold
    # add both circle but check if it's already in existing peaks
    i = (area / (area1 + area2)) <= threshold
    c1 = circles1[i,:]; c2 = circles2[i,:]
    peaks = [list(c) for c in c1 if (str(c[:2]) not in true_peaks) or \
                       (str(c[:2]) in true_peaks and \
                            c[3] > true_peaks[str(c[:2])][3])]
    update_peaks(true_peaks, peaks, [])

    peaks = [list(c) for c in c2 if (str(c[:2]) not in true_peaks) or \
                       (str(c[:2]) in true_peaks and \
                            c[3] > true_peaks[str(c[:2])][3])]
    update_peaks(true_peaks, peaks, [])

    # if overlap is greater than threshold
    # pick circles with greater peak values
    # check if rejected circle is in peaks.
    # if yes, delete it
    i = (area / (area1 + area2)) > threshold
    c1 = circles1[i,:]; c2 = circles2[i,:]            
    i = c1[:,3] > c2[:,3]
    tmp = c1[i]
    peaks = [list(c) for c in tmp if (str(c[:2]) not in true_peaks) or \
                  (str(c[:2]) in true_peaks and c[3] > true_peaks[str(c[:2])][3])]
    rejects = c1[~i].tolist()
    update_peaks(true_peaks, peaks, rejects)

    i = c2[:,3] > c1[:,3]
    tmp = c2[i]
    peaks = [list(c) for c in tmp if (str(c[:2]) not in true_peaks) or \
                  (str(c[:2]) in true_peaks and c[3] > true_peaks[str(c[:2])][3])]
    rejects = c2[~i].tolist()
    update_peaks(true_peaks, peaks, rejects)


def detect_blobs(image, sigma, k, num_layers, num_oct, threshold):
    """
    Given an image and user defined parameters, find blobs within the image
    
    Inputs:
    - image: 2D or 3D image array of original image
    - sigma: intial sigma value
    - k: scaling factor
    - num_layers: int; number of DoG layers within each octave
    - num_oct: int; number of octaves in scale space
    - threshold: value used for non-maximum suppression

    Outputs:
    - centers of blobs found in image
    """
    if np.ndim(image) == 3:
        gray = color.rgb2gray(image)
    elif np.ndim(image) == 2:
        gray = utils.intensity_spread(image, 2)
    else:
        raise ValueError("Accept 2D or 3D input image only")

    space, sigmas = build_scale_space(gray, sigma, k, num_layers, num_oct)
    peaks = find_peaks(space, sigmas, threshold)
    return peaks


# driver function
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-s", "--sigma", required=True, help="Initalize sigma value")
    ap.add_argument("-k", "--scale", required=True, help="Scaling sigma")
    ap.add_argument("-n", "--num_layers", required=True, 
                    help="number of layers in each octave in scale space")
    ap.add_argument("-N", "--num_oct", required=True, 
                    help="number of octaves in scale space")
    ap.add_argument("-t", "--threshold", required=True,
                    help="threshold used for non-max suppression")
    ap.add_argument("-o", "--output", required=True,
                    help="name of output image")

    args = ap.parse_args()
    path = args.image
    sigma = float(args.sigma)
    k = float(args.scale)
    num_layers = int(args.num_layers)
    num_oct = int(args.num_oct)
    threshold = float(args.threshold)
    output = args.output

    image = io.imread(path)
    image = color.rgb2gray(image)

    start = time.time()
    peaks = detect_blobs(image, sigma, k, num_layers, num_oct, threshold)
    projected_peaks = []
    for i in range(num_oct):
        img_scale = 2**i
        octave_peaks = list(peaks[i].values())
        projected_peaks += project_points(image, img_scale, octave_peaks)
    print(time.time() - start)

    result = image.copy()
    result = color.gray2rgb(result)
    utils.draw_circles(result, projected_peaks)
    io.imsave(output, result)

if __name__ == "__main__":
    main()
