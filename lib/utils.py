import numpy as np

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
