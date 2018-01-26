import os
import cv2
import numpy as np
from scipy import optimize
from scipy import misc
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import random
import math
from lib import image, model, utils, viz

# [Aligning Phase] Maximum acceptable distance when calculating matches of 
# ORB features.
MAXIMUM_ACCEPTABLE_HAMMING_DISTANCE = 20
CAMERA_CORRECTION_LAMBDA = 1.0
CAMERA_CORRECTION_BUCKET = 5
# [Depth Map Phase] Stdandard deviations of Gaussian blurring kernels when
# estimating blurry-ness of pixels in an image.
MAX_BLURRY_RADIUS = 100.0
BLURRY_STACK_SIZE = 93
INTEGRATION_STEPS = 29
IMG_SUM_SIZE = 25
# [Depth Map Phase] Window size to calculate difference between blurred
# refocused image and original image.
# BLUR_EST_WINDOW_SIZE = 5  # Chem
BLUR_EST_WINDOW_SIZE = 20 # Camera
# VERY_FAR = 3700 # Chem
VERY_FAR = 700  # Camera
# [Smoothing Phase] When smoothing the image, we first need to remove 
# extreme outliers. In percenage. 0.8 means remove the highest 0.8% and the 
# lowest 0.8%.
DEPTH_SMOOTH_TRUNC_THRESHOLD = 1.0
# [Smoothing Phase] Denoising strength.
DEPTH_SMOOTH_DENOISE_STRENGTH = 10
# [Smoothing Phase] Inpainting size.
DEPTH_SMOOTH_INPAINT_SIZE = 30 # 10

CHEM_DATA_WD = [3620, 3630, 3633.726, 3640, 3650, 3660, 3670]
# PIXEL_SIZE = 0.055  # Chem 
APS_C_SENSOR_SIZE = (23.5, 15.6) # Camera, in mm
FOCAL_LENGTH = 35 # Camera, in mm
FOREGROUND_RECT = [1327, 829, 3391, 2555]

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

  wds = image.extract_wd_from_exif(\
      [blurry_path + filename for filename in img_filenames])
  print(wds)
  return imgs, benchmark_img, wds

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
  cropbox = utils.calc_max_incribed_rect(translations, \
      [orig_benchmark_img.shape[1], orig_benchmark_img.shape[0]])
  cropped_imgs = [ \
      img[cropbox[0, 0] : cropbox[0, 1], \
          cropbox[1, 0] : cropbox[1, 1]] \
      for img in aligned_imgs]
  cropped_benchmark_img = orig_benchmark_img[ \
      cropbox[0, 0] : cropbox[0, 1], \
      cropbox[1, 0] : cropbox[1, 1]]
  return cropped_imgs, cropped_benchmark_img
  
def estimate_blurry_maps(imgs, scale, benchmark, mode='gaussian', filter_param=[]):
  print('Estimating blurry-ness of image stacks...')
  if imgs[0].ndim == 3:
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    benchmark = cv2.cvtColor(benchmark, cv2.COLOR_BGR2GRAY)

  blurry_stack = None
  if mode == 'gaussian':
    blurry_stack = [np.array(\
        cv2.GaussianBlur(benchmark, (7, 7), abs(var) * 0.1) \
            if var != 0 else benchmark, \
        np.float32) \
        for var in filter_param]
  elif mode == 'softmax':
    blurry_stack = [image.filter_by_energy(benchmark, filter) \
        for filter in utils.gen_disk_filters(filter_param)]
  elif mode == 'disk':
    blurry_stack = [cv2.filter2D(benchmark, -1, filter) \
        for filter in utils.gen_disk_filters(filter_param)]

  boxsize = 2 * math.ceil(BLUR_EST_WINDOW_SIZE * scale) + 1
  all_ssd_stack = np.array([[ \
      np.array(cv2.GaussianBlur(np.square(blur - img), \
          (boxsize, boxsize), 0.0), np.float32) \
      for blur in blurry_stack] \
      for img in imgs])
  return all_ssd_stack

def calc_energy(a, wds, dists, radii, ssds):
  n = len(wds)
  m = len(dists)
  total = 0
  for i in range(m):
    splines = [interpolate.interp1d(radii / a + 1, ssds[i][j, :], \
        kind='linear', fill_value='extrapolate') for j in range(n)]
    total += sum(splines[j](dists[i] / wds[j]) for j in range(n))
  return total

def correct_cam_params(ref_wds, radii, blurry_stacks, samples):
  print('Correcting camera parameters...')
  blurry_stacks = np.array(blurry_stacks, np.float32)

  n = len(ref_wds)
  bucket = CAMERA_CORRECTION_BUCKET
  lmd = CAMERA_CORRECTION_LAMBDA
  min_wd = np.min(ref_wds) * 0.9
  max_wd = np.max(ref_wds) * 1.1
  max_a = max(radii) * max_wd / (max_wd - min_wd)
  init = np.array([20] + [np.mean(ref_wds)] * (n + bucket))

  num_batch = len(samples) // bucket
  result_a = np.zeros([num_batch], np.float32)
  result_wds = np.zeros([num_batch, n], np.float32)
  for i in range(num_batch):
    if (i + 1) % 10 == 0:
      print('\tOptimizing batch #%d/#%d...' % (i + 1, num_batch))
    index_list = [bucket * i + j for j in range(bucket)]
    ssds = [blurry_stacks[:, :, samples[index, 1], samples[index, 0]] \
        for index in index_list]

    energy_func = lambda x: calc_energy(\
        x[0], x[1 : n + 1], x[n + 1 : ], radii, ssds) \
        + np.square(ref_wds - x[1 : n + 1]).sum() * lmd
    sol = optimize.minimize(energy_func, init, \
        bounds=[(1e-3, max_a)] + [(min_wd, max_wd)] * (n + bucket), \
        tol=0.01)
    
    result_a[i] = sol.x[0]
    result_wds[i, :] = sol.x[1 : n + 1]
    init[: n + 1] = sol.x[: n + 1]
  avg_a = np.mean(result_a)
  std_a = np.std(result_a)
  avg_wds = np.mean(result_wds, axis=0)
  std_wds = math.sqrt(np.mean([np.dot(wd, wd) for wd in result_wds]) \
      - np.dot(avg_wds, avg_wds))
  return avg_a, std_a, avg_wds, std_wds

def estimate_distances(wds, radii, blurry_stacks, mask, focused_img, scale):
  print('Estimating distances of foreground pixels...')
  blurry_stacks = np.array(blurry_stacks, np.float32)
  sum_boxsize = 2 * math.ceil(IMG_SUM_SIZE * scale) + 1
  sum_img = utils.sum_box_filter(focused_img, sum_boxsize)

  H = blurry_stacks.shape[2]
  W = blurry_stacks.shape[3]
  min_wd = np.min(wds)
  max_wd = np.max(wds)
  init = np.mean(wds)

  depth_map = VERY_FAR * np.ones([H, W], np.float32)
  conf_map = np.zeros([H, W], np.float32)
  total_pixels = np.count_nonzero(mask)
  count = 0
#   samples = [[700, 858], [701, 858], [701, 859], [702, 859], [783, 685]]
  for y in range(H):
    for x in range(W):
#   for p in samples:
#       y, x = p[1], p[0]
      if mask[y, x] == 0:
        continue
      count += 1
      if count % 1000 == 0:
        print('\tMinimizing integrations for #%d/%d...' % (count, total_pixels))

      ssd = blurry_stacks[:, :, y, x]
      
      spline = interpolate.RectBivariateSpline(wds, radii, ssd, kx=1, ky=1)
      energy_func = lambda d: sum(spline(wd, d / wd)[0, 0] \
          for wd in np.linspace(min_wd, max_wd, INTEGRATION_STEPS))
      sol = optimize.minimize_scalar(energy_func, \
          bounds=[min_wd, max_wd], method='bounded', tol=1.0)
      depth_map[y, x] = sol.x

    #   plot_x = np.linspace(min_wd, max_wd, num=100)
    #   plot_y = [energy_func(x) for x in plot_x]
    #   plt.plot(plot_x, plot_y)
    #   plt.savefig('./doll_data_out/sampleplot_%d_%d.png' % (x, y))
    #   plt.clf()
    #   viz.visualize_grid_map(ssd, wds, radii, \
    #       outpath='./doll_data_out/%d_%d_%.2f.png' % (x, y, sol.x))

      conf = misc.derivative(energy_func, sol.x, dx=2.0, n=2, order=5) / sol.fun
      conf_map[y, x] = np.log(conf / sum_img[y, x] + 1.0)

    #   print('Point (%d, %d): d=%.2f, conf=%.4f, energy=%.2f' % \
    #       (x, y, depth_map[y, x], conf_map[y, x], sol.fun))

      if count % 100000 == 0:
        viz.normalize_and_draw(depth_map, './doll_data_out/raw_depth_map.jpg', 0)
        viz.normalize_and_draw(conf_map, './doll_data_out/conf_map.jpg', 0)
        model.output_ply_file(depth_map, focused_img, \
            './doll_data_out/raw_model.ply', \
            sensor_size=APS_C_SENSOR_SIZE, focal_length=FOCAL_LENGTH, \
            break_thresh=5)
  return depth_map, conf_map

# Smooth the depth map.
def smooth_depth_map(depth, conf, percentile, fore_mask):
  print('Smoothing depth map with %d-th percentile...' % (percentile))

  # Mark pixels with confidence < x percentile as not credible in mask, so that
  # we can inpaint these areas from high-confidence pixels.
  conf_thresh = np.percentile(conf, percentile)
  mask = np.zeros(depth.shape, np.uint8)
  mask[(conf < conf_thresh) & (fore_mask == 1)] = 1

  output_depth = np.array(depth)
  output_depth[mask == 1] = VERY_FAR
  viz.normalize_and_draw(output_depth, './doll_data_out/masked_depth.jpg', 0)

  norm_depth, scale, shift = utils.normalize(depth, 0, 255.0)
  smooth_depth = np.array(norm_depth, np.uint8)
  smooth_depth = cv2.inpaint(smooth_depth, mask, \
      DEPTH_SMOOTH_INPAINT_SIZE, cv2.INPAINT_TELEA)
  viz.normalize_and_draw(smooth_depth, './doll_data_out/noisy_depth.jpg', 0)  
  smooth_depth = cv2.fastNlMeansDenoising(smooth_depth, \
      h=DEPTH_SMOOTH_DENOISE_STRENGTH)
  smooth_depth = np.array(smooth_depth, np.float32)
  smooth_depth *= scale
  smooth_depth += shift
  return smooth_depth

def main_chem():
  imgs = read_chem_images('./chem_data/')
  imgs, _ = align_images(imgs, np.array(imgs[3]))
  wds = CHEM_DATA_WD
  focused_img = image.refocus_image(imgs)
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
  blurry_radius = MAX_BLURRY_RADIUS * scale
  radii = np.linspace(-blurry_radius, blurry_radius, BLURRY_STACK_SIZE)
  imgs, _, ref_wds = read_camera_images(
      './doll_data/f4.0/', None, resize_factor=scale)

  benchmark_img = imgs[5]
  original_size = benchmark_img.shape
  imgs, benchmark_img = align_images(imgs, benchmark_img)
  benchmark_img = image.refocus_image(imgs)
  cv2.imwrite('./doll_data_out/benchmark_cropped.jpg', benchmark_img)

  #(95, 116, 900, 620)
  mask = image.extract_foreground(benchmark_img, \
      np.array(FOREGROUND_RECT) * scale)
  cv2.imwrite('./doll_data_out/foreground.jpg', mask * 255)

  all_blurry_stacks = estimate_blurry_maps(imgs, scale, benchmark_img, \
      mode='softmax', filter_param=radii)

  orb = cv2.ORB_create()
  benchmark_features = orb.detectAndCompute(benchmark_img, mask=None)
  samples = [f.pt for f in benchmark_features[0]]
  random.shuffle(samples)
  samples = np.array(samples, np.uint32)

  a, std_a, wds, std_wds = correct_cam_params(ref_wds, radii, \
      all_blurry_stacks, samples=samples)
  print('Estimated a = %f with standard variation %.2f' % (a, std_a))
  print('Corrected wds = %s with standard variation %.2f' % (wds, std_wds))

  radii = np.array(radii / a + 1, np.float32)
  depth_map, conf_map = estimate_distances(wds, radii, all_blurry_stacks, \
      mask, benchmark_img, scale)
  plt.hist(depth_map[mask == 1], bins=300, range=(400, 660))
  plt.savefig('./doll_data_out/depth_dist.jpg')
  plt.clf()
  plt.hist(conf_map[mask == 1], bins=300)
  plt.savefig('./doll_data_out/conf_dist.jpg')
  plt.clf()
  viz.normalize_and_draw(depth_map, './doll_data_out/raw_depth_map.jpg', 0)
  viz.normalize_and_draw(conf_map, './doll_data_out/conf_map.jpg', 0)
  model.output_ply_file(depth_map, benchmark_img, \
      './doll_data_out/raw_model.ply', \
      sensor_size=APS_C_SENSOR_SIZE, focal_length=FOCAL_LENGTH, \
      break_thresh=5)
  np.save('./doll_data_out/depth_map.npy', depth_map)
  np.save('./doll_data_out/conf_map.npy', conf_map)

  depth_map = np.load('./doll_data_out/depth_map.npy')
  conf_map = np.load('./doll_data_out/conf_map.npy')
  benchmark_img = cv2.imread('./doll_data_out/benchmark_cropped.jpg')
  mask = image.extract_foreground(benchmark_img, \
      np.array(FOREGROUND_RECT) * scale)
  depth_map[mask == 0] = VERY_FAR
  smooth_depth = smooth_depth_map(depth_map, conf_map, 90, mask)
  smooth_depth[mask == 0] = VERY_FAR
  viz.normalize_and_draw(smooth_depth, './doll_data_out/depth_map.jpg', 0)
  viz.normalize_and_draw(conf_map, './doll_data_out/conf_map.jpg', 0)

  model.output_ply_file(smooth_depth, benchmark_img, \
      './doll_data_out/model.ply', \
      sensor_size=APS_C_SENSOR_SIZE, focal_length=FOCAL_LENGTH, \
      break_thresh=5)

if __name__ == '__main__':
  main_doll()
