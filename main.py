import os
import math
import cv2
import numpy as np
import random
from scipy import stats
from scipy import optimize
from scipy import interpolate
import pyexifinfo as pei
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as lines
from lib import model, utils, viz

# [Aligning Phase] Maximum acceptable distance when calculating matches of 
# ORB features.
MAXIMUM_ACCEPTABLE_HAMMING_DISTANCE = 20
# [Refocusing Phase] Window size when calculating variance for surrounding
# patch of a pixel.
VAR_EST_WINDOW_SIZE = 15
# [Depth Map Phase] Stdandard deviations of Gaussian blurring kernels when
# estimating blurry-ness of pixels in an image.
BLURRY_STACK_RADII = np.arange(-20, 21)
# [Depth Map Phase] Window size to calculate difference between blurred
# refocused image and original image.
# BLUR_EST_WINDOW_SIZE = 5  # Chem
BLUR_EST_WINDOW_SIZE = 25 # Camera
# [Depth Map Phase] Relationship between blurry-ness and working distance can
# be estimated by a linear formula b = a * (WD - c). where a is only related to
# device intrinsic params and c is related to distance of a point. When
# estimating a across pixels, we only pick those with high confidence.
DEPTH_SLOPE_EST_CONFIDENCE = 90
DEPTH_SAMPLING_RATIO = 0.01
# VERY_FAR = 3700 # Chem
VERY_FAR = 0.7  # Camera
# [Smoothing Phase] When smoothing the image, we first need to remove 
# extreme outliers. In percenage. 0.8 means remove the highest 0.8% and the 
# lowest 0.8%.
DEPTH_SMOOTH_TRUNC_THRESHOLD = 1.0
# [Smoothing Phase] Denoising strength.
DEPTH_SMOOTH_DENOISE_STRENGTH = 10
# [Smoothing Phase] Inpainting size.
DEPTH_SMOOTH_INPAINT_SIZE = 15 # 10
DEPTH_SMOOTH_GAUSSIAN_BLUR_SIGMA = 4.0 # 3.0
EXPO_CONST = 5.0

CHEM_DATA_WD = [3620, 3630, 3633.726, 3640, 3650, 3660, 3670]
# PIXEL_SIZE = 0.055  # Chem 
PIXEL_SIZE = 0.0001 # Camera

def read_and_preprocess(filename, resize_factor=1.0, grayscale=True):
  img = cv2.imread(filename, \
      cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
  img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
  return img

def read_chem_images(path, resize_factor=1.0):
  print('Reading and pre-processing images...')
  img_filenames = os.listdir(path)
  imgs = [read_and_preprocess(path + filename, resize_factor) \
      for filename in img_filenames]
  return imgs

def read_camera_images(blurry_path, benchmark_path, resize_factor=1.0):
  print('Reading and pre-processing images...')

  img_filenames = os.listdir(blurry_path)
  imgs = [read_and_preprocess(blurry_path + filename, \
      resize_factor, grayscale=False) \
      for filename in img_filenames]
  benchmark_img = read_and_preprocess(benchmark_path, \
      resize_factor, grayscale=False) \
      if benchmark_path is not None else None

  exifs = [pei.get_json(blurry_path + filename)[0] \
      for filename in img_filenames]
  wds = [float(exif['MakerNotes:FocusDistance'][:-2]) for exif in exifs]
  print(wds)
  return imgs, benchmark_img, wds

def calc_max_incribed_square(trans, shape):
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

def align_images(imgs, benchmark_img):
  print('Aligning images...')
  scale = 800.0 / benchmark_img.shape[1]
  orig_imgs = imgs
  orig_benchmark_img = benchmark_img
  benchmark_img = cv2.resize(benchmark_img, (0, 0), fx=scale, fy=scale)
  imgs = [cv2.resize(img, (0, 0), fx=scale, fy=scale) for img in imgs]
  orb = cv2.ORB_create()
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  
  benchmark_features = orb.detectAndCompute(benchmark_img, mask=None)
  translations = []
  aligned_imgs = []
  for (orig_img, img) in zip(orig_imgs, imgs):
    # A tuple of (keypoints, descriptors).
    features = orb.detectAndCompute(img, mask=None)
    # Match features between img and benchmark img, and filter out those with
    # Hamming distance less than the threshold.
    matches = [match for match in \
        matcher.match(features[1], benchmark_features[1]) \
        if match.distance < MAXIMUM_ACCEPTABLE_HAMMING_DISTANCE]
    # Estimate rigid transformation from matched points.
    kps_selected = np.array([features[0][match.queryIdx].pt \
        for match in matches])
    kps_benchmark_selected = \
        np.array([benchmark_features[0][match.trainIdx].pt \
        for match in matches])
    mat = cv2.estimateRigidTransform(kps_selected, kps_benchmark_selected, \
        fullAffine=False)
    mat[:, 2] /= scale
    # Warp the image.
    translations += [mat]
    aligned_imgs += [cv2.warpAffine(orig_img, mat, dsize=None, dst=None)]
  
  # Crop images to the maximum square enclosed in all transformed images.
  cropbox = calc_max_incribed_square(translations, \
      [orig_benchmark_img.shape[1], orig_benchmark_img.shape[0]])
  cropped_imgs = [ \
      img[cropbox[0, 0] : cropbox[0, 1], \
          cropbox[1, 0] : cropbox[1, 1]] \
      for img in aligned_imgs]
  cropped_benchmark_img = orig_benchmark_img[ \
      cropbox[0, 0] : cropbox[0, 1], \
      cropbox[1, 0] : cropbox[1, 1]]
  return cropped_imgs, cropped_benchmark_img

def get_focused_image(imgs):
  print('Generating Refocused image...')
  vars = [utils.variance_box_filter(img, VAR_EST_WINDOW_SIZE) for img in imgs]
  max_indices = np.argmax(vars, axis=0)
  focused_img = np.array(imgs[0], np.uint8)
  for y in range(focused_img.shape[0]):
    for x in range(focused_img.shape[1]):
      if focused_img.ndim == 2:
        focused_img[y, x] = imgs[max_indices[y, x]][y, x]
      else:
        focused_img[y, x, :] = imgs[max_indices[y, x]][y, x, :]
  return focused_img

def filter_with_softmax(img, filter):
  exp_sig = np.exp(img * EXPO_CONST / 255.0)
  exp_sig = cv2.filter2D(exp_sig, -1, filter)
  sig = np.log(exp_sig) * 255.0 / EXPO_CONST
  return sig
  
def estimate_blurry_maps(imgs, benchmark, mode='gaussian', filter_param=[]):
  print('Estimating blurry-ness of image stacks...')
  if imgs[0].ndim == 3:
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    benchmark = cv2.cvtColor(benchmark, cv2.COLOR_BGR2GRAY)

  blurry_stack = None
  if mode == 'gaussian':
    blurry_stack = [np.array(\
        cv2.GaussianBlur(benchmark, (7, 7), abs(var) * 0.1) \
            if var !=0 else benchmark, \
        np.float32) \
        for var in filter_param]
  elif mode == 'softmax':
    blurry_stack = [filter_with_softmax(benchmark, filter) \
        for filter in utils.gen_disk_filters(filter_param)]
  elif mode == 'disk':
    blurry_stack = [cv2.filter2D(benchmark, -1, filter) \
        for filter in utils.gen_disk_filters(filter_param)]

  boxsize = (BLUR_EST_WINDOW_SIZE, BLUR_EST_WINDOW_SIZE)
  all_ssd_stack = np.array([[
      np.array(cv2.boxFilter(np.square(blur - img), ddepth=-1,
          ksize=boxsize, normalize=False), np.float32)
      for blur in blurry_stack] \
      for img in imgs])
  return all_ssd_stack

def estimate_a(wds, radii, blurry_stacks, samples):
  print('Estimating ratio of blurriness and working distance...')
  blurry_stacks = np.array(blurry_stacks, np.float32)

  inv_wds = np.flipud(1.0 / np.array(wds, np.float32))
  blurry_stacks = np.flipud(blurry_stacks)
  min_wd = np.min(wds)
  max_wd = np.max(wds)
  min_rd = np.min(radii)
  max_rd = np.max(radii)
  min_a = max(radii) * max_wd / (min_wd - max_wd)
  init = [-40, np.mean(wds)]

  # Second dimension is [y, x, a, d, energy]
  sample_results = np.zeros((len(samples), 5))
  for i in range(len(samples)):
    if (i + 1) % 10 == 0:
      print('\tMinimizing integrations for sample %d/%d...' % \
          (i + 1, len(samples)))
    y = samples[i][1]
    x = samples[i][0]
    ssd = blurry_stacks[:, :, y, x]
    spline = interpolate.RectBivariateSpline(inv_wds, radii, ssd, kx=1, ky=1)

    # x = [a, d], f = \int_{w_min}^{w_max}spline(1/w, a(d/w-1))
    sum_func = lambda x: sum(spline(inv_wd, x[0] * (x[1] * inv_wd - 1)) \
        for inv_wd in np.linspace(1.0 / max_wd, 1.0 / min_wd, num=100))
    sol = optimize.minimize(sum_func, init, \
        bounds=[(min_a, 0.0), (min_wd, max_wd)])
    sample_results[i, :] = [y, x, sol.x[0], sol.x[1], sum_func(sol.x)]

    # if i % 10 == 0:
    #   viz.visualize_grid_map(ssd, inv_wds, radii, \
    #       outpath='./chem_data_out/%d_%d_%d_%.2f_%.2f.png' \
    #           % (i+1, x, y, sol.x[0], sol.x[1]), \
    #       line=sol.x)

  viz.plot_scatter(sample_results[:, 2], sample_results[:, 4], \
      outpath='./doll_data_out/sample_fit.png')
  qualified = sample_results[:, 4] < np.percentile(sample_results[:, 4], 80)
  avg_a = np.average(sample_results[qualified, 2])
  std_a = np.std(sample_results[qualified, 2])
  print('a=%.2f, std=%.2f' % (avg_a, std_a))
  return avg_a, std_a

def estimate_distances(wds, radii, blurry_stacks, a, mask, focused_img):
  print('Estimating distances of foreground pixels...')
  blurry_stacks = np.array(blurry_stacks, np.float32)

  H = blurry_stacks.shape[2]
  W = blurry_stacks.shape[3]
  inv_wds = np.flipud(1.0 / np.array(wds, np.float32))
  blurry_stacks = np.flipud(blurry_stacks)
  min_wd = np.min(wds)
  max_wd = np.max(wds)
  min_rd = np.min(radii)
  max_rd = np.max(radii)
  init = np.mean(wds)

  depth_map = VERY_FAR * np.ones([H, W], np.float32)
  conf_map = np.zeros([H, W], np.float32)
  total_pixels = np.count_nonzero(mask)
  count = 0
  for y in range(H):
    for x in range(W):
      if mask[y, x] == 0:
        continue
      count += 1
      if count % 100 == 0:
        print('\tMinimizing integrations for #%d/%d...' % (count, total_pixels))

      ssd = blurry_stacks[:, :, y, x]
      spline = interpolate.RectBivariateSpline(inv_wds, radii, \
          ssd, kx=1, ky=1)
      # f = \int_{w_min}^{w_max}spline(1/w, a(d/w-1))
      sum_func = lambda d: sum(spline(inv_wd, a * (d * inv_wd - 1)) \
          for inv_wd in np.linspace(1.0 / max_wd, 1.0 / min_wd, num=100))
      sol = optimize.minimize_scalar(sum_func, \
          bounds=[min_wd, max_wd], method='bounded')
      depth_map[y, x] = sol.x
      conf_map[y, x] = sum_func(sol.x)
      if count % 10000 == 0:
        viz.normalize_and_draw(depth_map, './doll_data_out/depth_map.png', 0)
        viz.normalize_and_draw(conf_map, './doll_data_out/conf_map.png', 0)
        model.output_ply_file(depth_map, focused_img, \
            './doll_data_out/model.ply', pixel_size=PIXEL_SIZE)
  return depth_map, conf_map

# Smooth the depth map.
def smooth_depth_map(depth, conf, percentile):
  print('Smoothing depth map with %d-th percentile...' % (percentile))

  # Mark pixels with confidence < x percentile as not credible in mask, so that
  # we can inpaint these areas from high-confidence pixels.
  conf_thresh = np.percentile(conf, percentile)
  mask = np.zeros(depth.shape, np.uint8)
  mask[conf < conf_thresh] = 1
  depth_with_mask = np.array(depth, np.float32)
  depth_with_mask[mask == 1] = 0
  
  # Remove the extremge 0.8-th percentile from depth map, then apply a
  # 4-stage smoothing: Denoising, Inpainting, Denoinsing, Gaussian Blurring.
  norm_depth, scale = utils.normalize(depth, \
      DEPTH_SMOOTH_TRUNC_THRESHOLD, 255.0)
  smooth_depth = np.array(norm_depth, np.uint8)
  smooth_depth = cv2.fastNlMeansDenoising(smooth_depth, \
      h=DEPTH_SMOOTH_DENOISE_STRENGTH)
  smooth_depth = cv2.inpaint(smooth_depth, mask, \
      DEPTH_SMOOTH_INPAINT_SIZE, cv2.INPAINT_TELEA)
  smooth_depth = cv2.fastNlMeansDenoising(smooth_depth, \
      h=DEPTH_SMOOTH_DENOISE_STRENGTH)
  smooth_depth = np.array(smooth_depth, np.float32)
  smooth_depth *= scale
  smooth_depth = cv2.GaussianBlur(smooth_depth, (0, 0), \
      DEPTH_SMOOTH_GAUSSIAN_BLUR_SIGMA)
  return smooth_depth

def extract_foreground(img, rect):
  mask = np.zeros(img.shape[:2], np.uint8)
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)
  cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, \
      mode=cv2.GC_INIT_WITH_RECT)
  pos_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
  return pos_mask

def main_chem():
  imgs = read_chem_images('./chem_data/')
  imgs, _ = align_images(imgs, np.array(imgs[3]))
  wds = CHEM_DATA_WD
  focused_img = get_focused_image(imgs)
  cv2.imwrite('./chem_data_out/focused_image.tiff', focused_img)

  all_blurry_stacks = estimate_blurry_maps(imgs, focused_img, mode='gaussian',
      filter_param=BLURRY_STACK_RADII)
  
  orb = cv2.ORB_create()
  benchmark_features = orb.detectAndCompute(focused_img, mask=None)
  samples = np.array([f.pt for f in benchmark_features[0]], np.uint32)

  a, std = estimate_a(wds, BLURRY_STACK_RADII, \
      all_blurry_stacks, samples=samples)

  mask = np.ones(focused_img.shape)
  depth_map, conf_map = estimate_distances(wds, BLURRY_STACK_RADII, \
      all_blurry_stacks, a, mask, focused_img)

  # smooth_depth = smooth_depth_map(depth_map, conf_map, 60)
  viz.normalize_and_draw(depth_map, './chem_data_out/depth_map.tiff', 0)

  model.output_ply_file(depth_map, focused_img, \
      './chem_data_out/model.ply', pixel_size=PIXEL_SIZE)

def main_doll():
  scale = 0.2
  imgs, benchmark_img, wds = read_camera_images(
      './doll_data/single/', './doll_data/benchmark.JPG', resize_factor=scale)
  imgs, benchmark_img = align_images(imgs, benchmark_img)
  
  cv2.imwrite('./doll_data_out/benchmark_cropped.jpg', benchmark_img)
  all_blurry_stacks = estimate_blurry_maps(imgs, benchmark_img, \
      mode='softmax', filter_param=BLURRY_STACK_RADII)

  orb = cv2.ORB_create()
  benchmark_features = orb.detectAndCompute(benchmark_img, mask=None)
  samples = np.array([f.pt for f in benchmark_features[0]], np.uint32)

  a, std = estimate_a(wds, BLURRY_STACK_RADII, \
      all_blurry_stacks, samples=samples)
  print('Estimated a = %f with standard variation %f' % (a, std))

  mask = extract_foreground(benchmark_img, (95, 116, 900, 620))

  depth_map, conf_map = estimate_distances(wds, BLURRY_STACK_RADII, \
      all_blurry_stacks, a, mask, benchmark_img)
  # smooth_depth = smooth_depth_map(depth_map, conf_map, 60)
  viz.normalize_and_draw(depth_map, './doll_data_out/depth_map.jpg', 0)

  model.output_ply_file(depth_map, benchmark_img, \
      './doll_data_out/raw_model.ply', \
      pixel_size = PIXEL_SIZE / scale)

if __name__ == '__main__':
  main_doll()