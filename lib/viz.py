import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as lines
import numpy as np
from scipy import interpolate
from .utils import normalize, hist_equalize

def normalize_and_draw(img, outpath, norm_percentile):
  """
    Normalize a single-channel image to [0, 255] with outliers removed. Then
    output the image to specified path.

    Args:
        img: Input image as a numpy array.
        outpath: Path to output the image.
        norm_percentile: top- and bottom-percentile of pixel values to be
                         truncated as outliers.
  """
  outimg, _, _ = normalize(img, norm_percentile, 255.0)
  outimg = np.array(outimg, np.uint8)
  cv2.imwrite(outpath, outimg)

def visualize_grid_map(values, x, y, outpath=None, line=None):
  """
    Visualize a 2-dimentional function f defined by values on grid points.
    Demonstrate the function via color map, using first-order Spline
    interpolation on grid points.

    Args:
        values: An n*m matrix defining function values. values[i, j] represents
                the function value f(yi, xj).
        x: A 1-d array of length m representing x-values of the grid points.
        y: A 1-d array of length n representing y-values of the grid points.
        outpath: Output path of the visualized image. If none, will
                 display the image on a new window.
        line: Optionally renders a fitted line on the image if specified. This
              param should be a 1-d array of length 2 (say [a, b]), representing
              the line as y=a*(bx-1).
  """
  spline = interpolate.RectBivariateSpline(x, y, values, kx=1, ky=1)
  fig = plt.figure()
  axes = fig.add_subplot(111)
  range_func = lambda a, l: np.linspace(np.min(a), np.max(a), l)
  xi = range_func(x, 100)
  yi = range_func(y, 100)
  zi = spline(xi, yi)
  ticks = range_func(zi, 100)
  axes.contourf(xi, yi, zi.transpose(), ticks, cmap=cm.jet)
  if line is not None:
    xdata = [np.min(x), np.max(x)]
    ydata = [line[0] * (line[1] * np.min(x) - 1.0), \
             line[0] * (line[1] * np.max(x) - 1.0)]
    axes.add_line(lines.Line2D(xdata, ydata, linewidth=2.0, color='k'))
  if outpath is None:
    plt.show()
  else:
    plt.savefig(outpath)
  plt.clf()

def plot_scatter(x, y, percentile=100, outpath=None):
  x = x[y < np.percentile(y, percentile)]
  y = y[y < np.percentile(y, percentile)]
  fig = plt.figure()
  axes = fig.add_subplot(111)
  axes.plot(x, y, '.')
  if outpath is None:
    plt.show()
  else:
    plt.savefig(outpath)
  plt.clf()
