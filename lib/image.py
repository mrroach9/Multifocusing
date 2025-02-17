import cv2
import numpy as np
import pyexifinfo as pei
from .utils import variance_box_filter

DEFAULT_EXPO_CONST = 5.0
DEFAULT_VAR_EST_WINDOW_SIZE = 15

def extract_wd_from_exif(filepaths):
  """
    Extracts working distance from EXIFs of a seris of images, specified by
    their file paths.

    Working distance is defined as the distance from the aperture to the
    closest surface of the object in sharp focus.

    Args:
        filepaths: List of paths for images to extract working distances.
    Returns:
        wds: A list of floating point numbers of working distances, in mm.
  """
  exifs = [pei.get_json(filepath)[0] for filepath in filepaths]
  wds = [float(exif['MakerNotes:FocusDistance'][:-2]) * 1000 for exif in exifs]
  return wds

def filter_by_energy(img, filter, expo_const=DEFAULT_EXPO_CONST):
  """
    Applies a filter to an image on exponential of pixel intensities rather than
    on pixel intensities.
    
    Pixel intensities are logarithmically propotional to the actual energy
    captured by camera, defined as EV (exposure value). The natural blending
    procedure of camera blurring is actually averaging the initial energy value,
    not the exposure values. The method follows the below formula to simulate
    camera blurring:
        I' = log(f * exp(aI))
    where * represents convolution, and a is the expo_const passed in.

    Args:
        img: Input image to be filtered
        filter: Input filter to be applied
        expo_const: A constant tuning the exponential transformation. The larger
                    this value is, the closer it is to max function.
    Returns:
        filtered_img: Filtered image.
  """
  exp_sig = np.exp(img * expo_const / 255.0)
  exp_sig = cv2.filter2D(exp_sig, -1, filter)
  sig = np.log(exp_sig) * 255.0 / expo_const
  return sig

def refocus_image(imgs, var_window_size=DEFAULT_VAR_EST_WINDOW_SIZE):
  """
    Generates a sharp focused image from a series of images, using CDAF
    algorithm.

    The input images must be geometrically aligned, with each object or point
    at least sharp focused in one of the image series.

    Args:
        imgs: List of images focused on different spatial planes, aligned.
        var_window_size: Size of box to calculate variance around each pixel.
    Returns:
        focused_img: A single image generated from input images, with each pixel
        extracted from the image with largest variance around that location.
  """
  print('Generating Refocused image...')
  vars = np.array([variance_box_filter(img, var_window_size) for img in imgs])
  max_indices = np.argmax(vars, axis=0)
  focused_img = np.array(imgs[0], np.uint8)
  for y in range(focused_img.shape[0]):
    for x in range(focused_img.shape[1]):
      if focused_img.ndim == 2:
        focused_img[y, x] = imgs[max_indices[y, x]][y, x]
      else:
        focused_img[y, x, :] = imgs[max_indices[y, x]][y, x, :]
  return focused_img

def extract_foreground(img, rect):
  """
    Extracts a continuous area as foreground of an image, bounded by a specified
    rectangle. This method uses GrabCut algorithm.

    Args:
        img: Input image for segmentation.
        rect: Input rectangle bounding the foreground area, in format
              [left, top, width, height].
    Returns:
        pos_mask: A 0-1 matrix with the same shape as input image, where 0
                  represents possible background pixels and 1 represents
                  foreground.
  """ 
  mask = np.zeros(img.shape[:2], np.uint8)
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)
  cv2.grabCut(img, mask, tuple(np.array(rect, np.uint32)), \
      bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_RECT)
  pos_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
  morph_kernel = np.ones((5, 5), np.uint8)
  pos_mask = cv2.morphologyEx(pos_mask, cv2.MORPH_OPEN, morph_kernel)
  pos_mask = cv2.morphologyEx(pos_mask, cv2.MORPH_CLOSE, morph_kernel)
  return pos_mask
