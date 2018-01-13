import cv2
import numpy as np
import skimage.morphology as skmorph

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
  return out, orig_scale / scale

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
  filters = [skmorph.disk(abs(r)) for r in radii]
  filters = [np.array(filter, np.float32) / np.sum(filter) \
      for filter in filters]
  return filters
