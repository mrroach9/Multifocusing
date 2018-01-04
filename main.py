import os
import math
import cv2
import numpy as np
import random
import gc
from scipy import stats
from scipy import optimize
from scipy import interpolate
from scipy import integrate
import pyexifinfo as pei
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as lines
import skimage.morphology as skmorph

# [Aligning Phase] Maximum acceptable distance when calculating matches of 
# ORB features.
MAXIMUM_ACCEPTABLE_HAMMING_DISTANCE = 20
# [Refocusing Phase] Window size when calculating variance for surrounding
# patch of a pixel.
VAR_EST_WINDOW_SIZE = 15
# [Depth Map Phase] Stdandard deviations of Gaussian blurring kernels when
# estimating blurry-ness of pixels in an image.
BLURRY_STACK_SIGMAS = [x * 0.2 for x in range(50)]
BLURRY_STACK_RADII = [x for x in np.arange(-40, 41)]
# [Depth Map Phase] Window size to calculate difference between blurred
# refocused image and original image.
BLUR_EST_WINDOW_SIZE = 25 #5
# [Depth Map Phase] Relationship between blurry-ness and working distance can
# be estimated by a linear formula b = a * (WD - c). where a is only related to
# device intrinsic params and c is related to distance of a point. When
# estimating a across pixels, we only pick those with high confidence.
DEPTH_SLOPE_EST_CONFIDENCE = 90
DEPTH_SAMPLING_RATIO = 0.01
DEPTH_EST_A_INITIAL = -1
DEPTH_EST_D_INITIAL = 0.6
DEPTH_EST_A_BOUND = (-120, 0)
DEPTH_EST_WEIGHT_COEFF = 0
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

CHEM_DATA_WD = [3633.726, 3670, 3660, 3650, 3640, 3630, 3620]
PIXEL_SIZE = 0.0001 #0.055

def read_and_preprocess(filename, resize_factor=1.0, grayscale=True):
  img = cv2.imread(filename, cv2.IMREAD_COLOR)
  img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

def read_chem_images(path):
  print('Reading and pre-processing images...')
  img_filenames = os.listdir(path)
  imgs = [cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE) \
      for filename in img_filenames]
  imgs = [cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor) \
      for img in imgs]
  return imgs

def read_camera_images(blurry_path, benchmark_path, resize_factor=1.0):
  print('Reading and pre-processing images...')

  img_filenames = os.listdir(blurry_path)
  imgs = [read_and_preprocess(blurry_path + filename, resize_factor) \
      for filename in img_filenames]
  benchmark_img = read_and_preprocess(benchmark_path, resize_factor)

  exifs = [pei.get_json(blurry_path + filename)[0] \
      for filename in img_filenames]
  wds = [float(exif['MakerNotes:FocusDistance'][:-2]) for exif in exifs]
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

def variance_box_filter(img, boxsize):
  img = np.array(img, np.float32)
  kernel =(boxsize, boxsize)
  EX_2 = np.square(cv2.boxFilter(img, ddepth=-1, ksize=kernel))
  E_X2 = cv2.boxFilter(img * img, ddepth=-1, ksize=kernel)
  return E_X2 - EX_2

def solve_distance_eq(wds, sigmas):
  x_list = np.array(wds, np.float32)
  y_orig_list = np.array(sigmas, np.float32)

  best_solution = (0, 0, 0, 0)
  critical_x = x_list[np.argmin(y_orig_list)]
  
  for x in [critical_x + 1e-6, critical_x - 1e-6]:
    y_list = [(2 * int(x_list[i] < x) - 1) * y_orig_list[i] \
        for i in range(len(y_orig_list))]
    a, b, r, p, stderr = stats.linregress(x_list, y_list)
    if abs(r) > abs(best_solution[2]):
      best_solution = (a, -b / a, abs(r), y_list)
  return best_solution

def normalize(depth, percentile, scale=None):
  out = np.array(depth, np.float32)
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

def normalize_and_draw(img, outpath, norm_percentile):
  outimg, _ = normalize(img, norm_percentile, 255.0)
  outimg = np.array(outimg, np.uint8)
  cv2.imwrite(outpath, outimg)

def gen_disk_filters(radii):
  filters = [skmorph.disk(abs(r)) for r in radii]
  filters = [np.array(filter, np.float32) / np.sum(filter) \
      for filter in filters]
  return filters

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
  vars = [variance_box_filter(img, VAR_EST_WINDOW_SIZE) for img in imgs]
  max_indices = np.argmax(vars, axis=0)
  focused_img = np.array(imgs[0], np.uint8)
  for y in range(focused_img.shape[0]):
    for x in range(focused_img.shape[1]):
      focused_img[y, x] = imgs[max_indices[y, x]][y, x]
  return focused_img

def filter_with_softmax(img, filter):
  exp_sig = np.exp(img * EXPO_CONST / 255.0)
  exp_sig = cv2.filter2D(exp_sig, -1, filter)
  sig = np.log(exp_sig) * 255.0 / EXPO_CONST
  return sig

def visualize_blurry_similarity(ssd_stack, wds, radii, outpath=None, line=None):
  spline = interpolate.RectBivariateSpline(wds, radii, \
      ssd_stack, kx=1, ky=1)
  fig = plt.figure()
  axes = fig.add_subplot(111)
  range_func = lambda a, l: np.linspace(np.min(a), np.max(a), l)
  xi = range_func(wds, 100)
  yi = range_func(radii, 100)
  zi = spline(xi, yi)
  ticks = range_func(zi, 100)
  axes.contourf(xi, yi, zi.transpose(), ticks, cmap=cm.jet)
  if line is not None:
    xdata = [np.min(wds), np.max(wds)]
    ydata = [line[0] * (line[1] * np.min(wds) - 1.0), \
             line[0] * (line[1] * np.max(wds) - 1.0)]
    axes.add_line(lines.Line2D(xdata, ydata, linewidth=2.0, color='k'))
  if outpath is None:
    plt.show()
  else:
    plt.savefig(outpath)
  
def estimate_blurry_maps(imgs, benchmark, wds):
  print('Estimating blurry-ness of image stacks...')
  blurry_stack = [filter_with_softmax(benchmark, filter) \
      for filter in gen_disk_filters(BLURRY_STACK_RADII)]

  boxsize = (BLUR_EST_WINDOW_SIZE, BLUR_EST_WINDOW_SIZE)
  all_ssd_stack = np.array([[
      np.array(cv2.boxFilter(np.square(blur - img), ddepth=-1,
          ksize=boxsize, normalize=False), np.float32)
      for blur in blurry_stack] \
      for img in imgs])
  return all_ssd_stack
  
  # blurry_maps = []
  # for img in imgs:
  #   diff_stack = [np.square(blur - img) for blur in blurry_stack]
  #   boxsize = BLUR_EST_WINDOW_SIZE
  #   ssd_stack = [cv2.boxFilter(diff_img, ddepth=-1, \
  #           ksize=(boxsize, boxsize), normalize=False) \
  #       for diff_img in diff_stack]
  #   min_indices = np.argmin(ssd_stack, axis=0)
  #   blurry_map = np.zeros(img.shape[:2], np.float32)
  #   for y in range(blurry_map.shape[0]):
  #     for x in range(blurry_map.shape[1]):
  #       blurry_map[y, x] = BLURRY_STACK_SIGMAS[min_indices[y, x]]
  #   blurry_maps += [blurry_map]
  # return blurry_maps

def estimate_depth(wds, blurry_maps):
  print('Estimating depth image...')
  blurry_maps = np.array(blurry_maps, np.float32)
  depth_map = np.zeros(blurry_maps[0].shape, np.float32)
  slope_map = np.zeros(depth_map.shape, np.float32)
  confidence_map = np.zeros(depth_map.shape, np.float32)
  for y in range(blurry_maps.shape[1]):
    if y % 10 == 0:
      print('\tProcesed: ' + str(y) + '/' + str(blurry_maps.shape[1]))
    for x in range(blurry_maps.shape[2]):
      solution = solve_distance_eq(wds, blurry_maps[:, y, x])
      blurry_maps[:, y, x] = solution[3]
      depth_map[y, x] = solution[1]
      slope_map[y, x] = solution[0]
      confidence_map[y, x] = solution[2]

  top_confidence = np.percentile(confidence_map, DEPTH_SLOPE_EST_CONFIDENCE)
  a_refined = np.mean(slope_map[confidence_map > top_confidence])
  depth_map_refined = np.mean(wds) - np.mean(blurry_maps, axis=0) / a_refined
  return depth_map_refined, confidence_map

def estimate_depth_new(wds, radii, blurry_stacks, samples=None):
  print('Estimating depth map...')
  blurry_stacks = np.array(blurry_stacks, np.float32)
  for idx, radius in enumerate(radii):
    blurry_stacks[:, idx, :, :] *= \
        math.exp(- DEPTH_EST_WEIGHT_COEFF * abs(radius))

  H = blurry_stacks.shape[2]
  W = blurry_stacks.shape[3]
  num_samples = int(W * H * DEPTH_SAMPLING_RATIO)
  inv_wds = np.flipud(1.0 / np.array(wds, np.float32))
  blurry_stacks = np.flipud(blurry_stacks)
  min_inv_wd = np.min(inv_wds)
  max_inv_wd = np.max(inv_wds)
  min_rd = np.min(radii)
  max_rd = np.max(radii)

  if samples is None:
    samples = [\
        [72, 189], [509, 479], [707, 393], [543, 825], [201, 949], \
        [70, 1003], [51, 616], [628, 113], [541, 1019], [667, 865]]

  # Second dimension is [y, x, a, d, energy]
  sample_results = np.zeros((len(samples), 5))
  for i in range(len(samples)):
    print('Minimizing integrations for sample %d/%d...' % (i, len(samples)))
    y = samples[i][1]
    x = samples[i][0]
    ssd = blurry_stacks[:, :, y, x]
    spline = interpolate.RectBivariateSpline(inv_wds, radii, \
        ssd, kx=1, ky=1)
    # x = [a, d], f = \int_{w_min}^{w_max}spline(1/w, a(d/w-1))
    integ_func = lambda x: \
        1e9 if x[0] * (x[1] * min_inv_wd - 1) < min_rd \
            or x[0] * (x[1] * min_inv_wd - 1) > max_rd \
            or x[0] * (x[1] * max_inv_wd - 1) < min_rd \
            or x[0] * (x[1] * max_inv_wd - 1) > max_rd \
        else integrate.quad(
            lambda inv_wd: spline(inv_wd, x[0] * (x[1] * inv_wd - 1)), \
                min_inv_wd, max_inv_wd)[0]
    sol = optimize.minimize(integ_func, \
        [DEPTH_EST_A_INITIAL, DEPTH_EST_D_INITIAL], \
        bounds=(DEPTH_EST_A_BOUND, [1.0 / max_inv_wd, 1.0 / min_inv_wd]))
    visualize_blurry_similarity(ssd, inv_wds, radii, \
        './doll_data_out/%d_%d_%d_%f_%f.png' % (i, x, y, sol.x[0], sol.x[1]), \
        line=sol.x)
    sample_results[i, :] = [y, x, sol.x[0], sol.x[1], integ_func(sol.x)]

  a_res = sample_results[:, 2]
  v_res = sample_results[:, 4]
  for p in [20, 50, 100]:
    a_sel = a_res[v_res < np.percentile(v_res, p)]
    v_sel = v_res[v_res < np.percentile(v_res, p)]
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(a_sel, v_sel, '.')
    plt.savefig('./doll_data_out/new_%d.png' % p)
#   print(a_res[v_res > np.percentile(v_res, 10)])
#   print(v_res[v_res > np.percentile(v_res, 10)])

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
  norm_depth, scale = normalize(depth, DEPTH_SMOOTH_TRUNC_THRESHOLD, 255.0)
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

# Output depth map into a PLY format 3D object, with benchmark image as
# vertex colors.
def output_ply_file(depth, img, outpath, pixel_size=PIXEL_SIZE):
  print('Outputting depth map to PLY file ' + outpath + '...')
  H = depth.shape[0]
  W = depth.shape[1]

  # Writing header.
  f = open(outpath, 'w+')
  f.write('ply\n')
  f.write('format ascii 1.0\n')
  f.write('element vertex %d\n' % (W * H))
  f.write('property float x\n')
  f.write('property float y\n')
  f.write('property float z\n')
  f.write('property uchar red\n')
  f.write('property uchar green\n')
  f.write('property uchar blue\n')
  f.write('element face %d\n' % (2 * (W - 1) * (H - 1)))
  f.write('property list uchar int vertex_index\n')
  f.write('end_header\n')
  for i in range(H):
    for j in range(W):
      # Our depth map is using a right-hand coordinate system:
      # X+: right, Y+: down, Z+: back.
      # Most 3D model viewer softwares use a different right-hand coordinate
      # system:
      # X+: right, Y+: up, Z+: front.
      y = - pixel_size * i# * depth[i, j]
      x = pixel_size * j# * depth[i, j]
      z = - depth[i, j]
      c = img[i, j]
      if img.ndim == 2:
        c = [c, c, c]
      # OpenCV loads image in BGR mode, but PLY assigns colors in RGB mode,
      # therefore we need to flip the color vector.
      f.write('%f %f %f %d %d %d\n' % (x, y, z, c[2], c[1], c[0]))
  # Print two triangles per pixel, vertices going counter-clockwise.
  # E.g. if the vertices of a pixel square are:
  #       1----2
  #       |    |
  #       3----4
  # We will output two triangles: (1, 3, 4) and (4, 2, 1).
  for i in range(H - 1):
    for j in range(W - 1):
      f.write('3 %d %d %d\n' % \
          (i * W + j, (i + 1) * W + j, (i + 1) * W + j + 1))
      f.write('3 %d %d %d\n' % \
          ((i + 1) * W + j + 1,  i * W + j + 1, i * W + j))
  f.flush()
  f.close()

def main_chem():
  imgs = read_chem_images('./chem_data/')
  imgs, _ = align_images(imgs, np.array(imgs[3]))
  focused_img = get_focused_image(imgs)
  cv2.imwrite('./chem_data_out/focused_image.tiff', focused_img)

  blurry_maps = estimate_blurry_maps(imgs, focused_img)
  depth_map, conf_map = estimate_depth(CHEM_DATA_WD, blurry_maps)
  smooth_depth = smooth_depth_map(depth_map, conf_map, 60)
  normalize_and_draw(smooth_depth, './chem_data_out/smooth_depth.tiff', 0)

  output_ply_file(smooth_depth, focused_img, './chem_data_out/model.ply')

def main_doll():
  scale = 0.2
  imgs, benchmark_img, wds = read_camera_images(
      './doll_data/blurred_stack/', './doll_data/benchmark.JPG',
      resize_factor=scale)
  imgs, benchmark_img = align_images(imgs, benchmark_img)
  cv2.imwrite('./doll_data_out/benchmark_cropped.jpg', benchmark_img)
  all_blurry_stacks = estimate_blurry_maps(imgs, benchmark_img, wds)

  orb = cv2.ORB_create()
  benchmark_features = orb.detectAndCompute(benchmark_img, mask=None)
  samples = np.array([f.pt for f in benchmark_features[0][:200]], np.uint32)
  print(samples)

  estimate_depth_new(wds, BLURRY_STACK_RADII, \
      all_blurry_stacks, samples=samples)
  # smooth_depth = smooth_depth_map(depth_map, conf_map, 60)
  # normalize_and_draw(smooth_depth, './doll_data_out/smooth_depth.jpg', 0)

  # output_ply_file(smooth_depth, benchmark_img, './doll_data_out/model.ply', \
  #     pixel_size = PIXEL_SIZE / scale)
  
if __name__ == '__main__':
  main_doll()