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
BLURRY_STACK_SIZE = 73
# VERY_FAR = 3700 # Chem
VERY_FAR = 700  # Camera
# [Smoothing Phase] Denoising strength.
DEPTH_SMOOTH_DENOISE_STRENGTH = 20
# [Smoothing Phase] Inpainting size.
DEPTH_SMOOTH_INPAINT_SIZE = 30 # 10

SMOOTH_WINDOW_HALFSIZE = 20
SMOOTH_DECAY = 0.98
CONF_DECAY = 0.90
DIST_COEFF = 10.0
INTEN_COEFF = 255.0 * 0.1
EXPAND_THRESH = 50.0

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

def read_camera_images(blurry_path, resize_factor=1.0):
  print('Reading and pre-processing images...')

  img_filenames = os.listdir(blurry_path)
  imgs = [read_and_preprocess(blurry_path + filename, \
      resize_factor, grayscale=False) \
      for filename in img_filenames]

  wds = image.extract_wd_from_exif(\
      [blurry_path + filename for filename in img_filenames])
  print(wds)
  return imgs, wds

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

  all_ssd_stack = np.array([[ \
      np.array(np.square(blur - img), np.float32) \
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

  H = blurry_stacks.shape[2]
  W = blurry_stacks.shape[3]
  min_wd = np.min(wds)
  max_wd = np.max(wds)
  init = np.mean(wds)

  depth_map = VERY_FAR * np.ones([H, W], np.float32)
  conf_map = - np.ones([H, W], np.float32)
  total_pixels = np.count_nonzero(mask)
  count = 0
  for y in range(H):
    for x in range(W):
      if mask[y, x] == 0:
        continue
      count += 1
      if count % 1000 == 0:
        print('\tMinimizing integrations for #%d/%d...' % (count, total_pixels))

      ssd = blurry_stacks[:, :, y, x]
      spline = interpolate.RectBivariateSpline(wds, radii, ssd, kx=1, ky=1)
      energy_func = lambda d: sum(spline(wd, d / wd)[0, 0] for wd in wds)
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

      conf = misc.derivative(energy_func, sol.x, dx=2.0, n=2, order=5)
      conf_map[y, x] = conf / sol.fun

      if count % 100000 == 0:
        output_results(depth_map, conf_map, mask, focused_img, \
            './doll_data_out/', raw=True, output_model=False)
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

def get_intensity_weight(img, y, x, win_hsize):
  intensity_weight = np.array(img[\
      y - win_hsize : y + win_hsize + 1, \
      x - win_hsize : x + win_hsize + 1])
  intensity_weight -= img[y, x]
  intensity_weight = np.exp(-intensity_weight ** 2 / INTEN_COEFF ** 2)
  return intensity_weight

def smooth_depth_map_new(depth, conf, mask, img):
  img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.float32)
  adj_conf = np.array(conf)
  adj_conf[mask == 1] = utils.hist_equalize(conf[mask == 1], (1e-3, 1.0))
  adj_conf[mask == 1] = np.exp(- 1.0 / adj_conf[mask == 1])
  adj_conf[mask == 0] = 0.0
  win_hsize = SMOOTH_WINDOW_HALFSIZE
  win_size = win_hsize * 2 + 1
  dist_filter = np.zeros([win_size, win_size], np.float32)
  for y in range(win_size):
    for x in range(win_size):
      dist_filter[y, x] = (y - win_hsize) ** 2 + (x - win_hsize) ** 2
  dist_filter = np.exp(-dist_filter / DIST_COEFF ** 2)
  depth = np.array(depth, np.float32)
  status_mask = np.array(mask, np.uint8)
  weight_map = np.zeros(depth.shape, np.float32)

  thresh = 1.0
  decay = SMOOTH_DECAY

  np.set_printoptions(precision=3)
  while True:
    thresh *= decay
    print('Smoothing with thresh=%.3f' % thresh)
    new_finalized = (adj_conf > math.exp(- 1.0 / thresh)) & (status_mask == 1)
    num_new_fin = np.count_nonzero(new_finalized)
    print('\tNewly finalized pixels: %d' % num_new_fin)
    if num_new_fin == 0:
      break
    status_mask[new_finalized] = 2
    for y in range(depth.shape[0]):
      for x in range(depth.shape[1]):
        if not new_finalized[y, x]:
          continue
        intensity_weight = get_intensity_weight(img, y, x, win_hsize)
        weight_patch = intensity_weight * dist_filter
        weight_map[y - win_hsize : y + win_hsize + 1, \
                   x - win_hsize : x + win_hsize + 1] += weight_patch
    new_expandable = (weight_map > EXPAND_THRESH) & (status_mask == 1)
    print('\tNewly expandable pixels: %d' % np.count_nonzero(new_expandable))

    for y in range(depth.shape[0]):
      for x in range(depth.shape[1]):
        if not new_expandable[y, x]:
          continue
        ready_patch = status_mask[\
            y - win_hsize : y + win_hsize + 1, \
            x - win_hsize : x + win_hsize + 1] == 2
        get_intensity_weight(img, y, x, win_hsize)
        conf_patch = adj_conf[\
            y - win_hsize : y + win_hsize + 1, \
            x - win_hsize : x + win_hsize + 1]
        depth_patch = depth[\
            y - win_hsize : y + win_hsize + 1, \
            x - win_hsize : x + win_hsize + 1]
        weight_patch = ready_patch * dist_filter * intensity_weight
        d = np.sum(weight_patch * conf_patch * depth_patch) / \
            np.sum(weight_patch * conf_patch)
        c = np.sum(weight_patch * conf_patch) / \
            np.sum(weight_patch)
        depth[y, x] = d
        adj_conf[y, x] = c * CONF_DECAY

  viz.normalize_and_draw(depth, './doll_data_out/unsmoothened_depth.jpg', 0)
  norm_depth, scale, shift = utils.normalize(depth, 0, 255.0)
  smooth_depth = np.array(norm_depth, np.uint8)  
  smooth_depth = cv2.fastNlMeansDenoising(smooth_depth, \
      h=DEPTH_SMOOTH_DENOISE_STRENGTH)
  smooth_depth = np.array(smooth_depth, np.float32)
  smooth_depth = (smooth_depth * scale) + shift
  return smooth_depth

def output_results(depth, conf, mask, img, folder, raw=True, output_model=True):
  depth_map_filename = 'raw_depth_map.jpg' if raw else 'depth_map.jpg'
  model_filename = 'raw_model.ply' if raw else 'model.ply'
  conf_map_filename = 'conf_map.jpg'
  depth_map_np_filename = 'raw_depth_map.npy' if raw else 'depth_map.npy'
  conf_map_np_filename = 'conf_map.npy'
  benchmark_img_filename = 'benchmark.jpg'
  foreground_img_filename = 'foreground.jpg'

  cv2.imwrite(folder + benchmark_img_filename, img)
  if mask is not None:
    cv2.imwrite(folder + foreground_img_filename, mask * 255)
  viz.normalize_and_draw(depth, folder + depth_map_filename, 0)
  np.save(folder + depth_map_np_filename, depth)

  if conf is not None and mask is not None:
    equalized_conf = np.array(conf)
    equalized_conf[mask == 1] = \
        utils.hist_equalize(conf[mask == 1], (1e-3, 1.0))
    equalized_conf[mask == 0] = 0.0
    viz.normalize_and_draw(equalized_conf, folder + conf_map_filename, 0)
    np.save(folder + conf_map_np_filename, conf)

  if output_model:
    model.output_ply_file(depth, img, folder + model_filename, \
        sensor_size=APS_C_SENSOR_SIZE, focal_length=FOCAL_LENGTH, \
        break_thresh=5)

def main_doll():
  scale = 0.3
  input_folder = './doll_data/f4.0/'
  output_folder = './doll_data_out/'
  blurry_radius = MAX_BLURRY_RADIUS * scale
  radii = np.linspace(-blurry_radius, blurry_radius, BLURRY_STACK_SIZE)
  imgs, ref_wds = read_camera_images(input_folder, resize_factor=scale)

  benchmark_img = imgs[len(imgs) // 2 - 1]
  original_size = benchmark_img.shape
  imgs, benchmark_img = align_images(imgs, benchmark_img)
  benchmark_img = image.refocus_image(imgs)

  mask = image.extract_foreground(benchmark_img, \
      np.array(FOREGROUND_RECT) * scale)

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
  output_results(depth_map, conf_map, mask, benchmark_img, \
      output_folder, raw=True, output_model=True)

  depth_map = np.load(output_folder + 'raw_depth_map.npy')
  conf_map = np.load(output_folder + 'conf_map.npy')
  benchmark_img = cv2.imread(output_folder + 'benchmark.jpg')
  mask = image.extract_foreground(benchmark_img, \
      np.array(FOREGROUND_RECT) * scale)
  depth_map[mask == 0] = VERY_FAR
  smooth_depth = smooth_depth_map_new(depth_map, conf_map, mask, benchmark_img)
  smooth_depth[mask == 0] = VERY_FAR
  output_results(smooth_depth, None, None, benchmark_img, \
      output_folder, raw=False, output_model=True)

if __name__ == '__main__':
  main_doll()
