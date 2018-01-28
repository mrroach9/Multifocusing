import cv2
import math
import numpy as np
from scipy import interpolate
import skimage.morphology as morph
import skimage.measure as measure

def normalize(mat, percentile=0, scale=None):
  """
    Normalizing a matrix from arbitrary range to [0, <specified scale>].
    Optionally truncate outliers -- top and bottom x% of the overall
    distribution.

    Args:
        mat: Input matrix.
        percentile: top- and bottom-percentile to be truncated as outliers. 0 if
                    not specified.
        scale: target upper bound of normalizing. If None, will not scale data.

    Returns:
        out: The normalized matrix with outliers truncated.
        inv_scale: inverse of actual factor of scaling applied. Multiplying out
                   by inv_scale will give the same range of the original matrix
                   (with zero-shifted and outliers truncated).
  """
  out = np.array(mat, np.float32)
  upper = np.percentile(out, 100 - percentile)
  lower = np.percentile(out, percentile)
  out[out < lower] = lower
  out[out > upper] = upper
  out -= lower
  orig_scale = np.max(out)
  if scale is None:
    scale = orig_scale
  out *= (scale / orig_scale)
  return out, orig_scale / scale, lower

def variance_box_filter(img, boxsize):
  """
    For each pixel in an image, calculate its variance of grayscale intensity
    of a square patch around it.

    Args:
        img: The input image, single-channel or 3-channel.
        boxsize: Size of the patch around each pixel to calculate variance.
                 Must be a positive odd integer.
    Returns:
        var: A single-channel matrix with the same dimension as the input image,
             with each cell storing the variance around the corresponding pixel
             in the image.
  """
  if img.ndim == 3:
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.float32)
  else:
    img = np.array(img, np.float32)
  kernel = (boxsize, boxsize)
  EX_2 = np.square(cv2.boxFilter(img, ddepth=-1, ksize=kernel))
  E_X2 = cv2.boxFilter(img * img, ddepth=-1, ksize=kernel)
  return E_X2 - EX_2

def gen_disk_filters(radii):
  """
    Generates disk filters for a list of specified radii.
    For a given number r, the disk filter is a 0-1 matrix A where
        A[c+i, c+j] = 1 if i^2+j^2 <= r^2, otherwise 0,
    where c is the center of the matrix.
    Radius r can be negative, in which case the result is identical to the disk
    filter of |r|.

    Args:
        radii: A list of disk radii.
    Returns:
        filters: A list of generated disk filters.
  """
  filters = []
  block = 4
  for r in radii:
    if abs(r) <= 1:
      filters += [np.identity(1, dtype=np.float32)]
      continue
    half_size = math.floor(abs(r)) + 1
    size = block * (2 * half_size + 1)
    f = np.zeros([size, size], np.float32)
    for i in range(size):
      for j in range(size):
        if (i + 0.5 - size / 2) ** 2 + (j + 0.5 - size / 2) ** 2 <= \
            block * block * r * r:
          f[i, j] = 1
    f = measure.block_reduce(f, (block, block), np.mean)
    if np.sum(f) > 0:
      f /= np.sum(f)
    filters += [f]
  return filters

def calc_max_incribed_rect(trans, shape):
  """
    For a given set of affine transformations (H0=I, H1, H2, ... Hn), and a 
    rectangular shape S, calculate the maximum rectangular that can be inscribed
    in Hi*S, with edges parallel to axes.

    Args:
        trans: A list of affine transformations, note that I is not included.
        shape: A rectangular to be transformed by trans.
    Returns:
        The maximum inscribed rectangular.
  """
  trans = trans + [np.array([[1, 0, 0], [0, 1, 0]], np.float32)]
  topleft = [tran.dot([0, 0, 1]) for tran in trans]
  topright = [tran.dot([shape[0] - 1, 0, 1]) for tran in trans]
  bottomleft = [tran.dot([0, shape[1] - 1, 1]) for tran in trans]
  bottomright = [tran.dot([shape[0] - 1, shape[1] - 1, 1]) for tran in trans]

  topedge = math.ceil(np.max(np.array(topleft + topright)[:, 1]))
  bottomedge = math.floor(np.min(np.array(bottomleft + bottomright)[:, 1]))
  leftedge = math.ceil(np.max(np.array(topleft + bottomleft)[:, 0]))
  rightedge = math.floor(np.min(np.array(topright + bottomright)[:, 0]))
  return np.array([[topedge, bottomedge], [leftedge, rightedge]], np.uint32)

def hist_equalize(data, bound):
  a = bound[0]
  r = bound[1] - bound[0]
  hist, edges = np.histogram(data, bins=1000)
  cdf = np.cumsum(hist) / np.sum(hist)
  x_val = (edges[:-1] + edges[1:]) / 2.0
  cdf_func = interpolate.interp1d(x_val, cdf)
  equalized_func = np.vectorize(lambda x: a if x < x_val[0] \
      else a + r if x > x_val[-1] \
      else a + r * cdf_func(x))
  return equalized_func(data)
