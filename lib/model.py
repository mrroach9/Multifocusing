def output_ply_file(depth, img, outpath, pixel_size):
  """
    Output depth map into a PLY format 3D object, with benchmark image as
    vertex colors.

    Args:
        depth: A 2-d matrix representing the depth of each pixel.
        img: The image representing the texture of the model. If it's 3-channel,
             the colors are BGR; if 1-channel, it represents the grayscale.
        outpath: The path to save the model to.
        pixel_size: Physical size of each pixel.
  """
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
      # Most of the 3D model viewer softwares use a different right-hand
      # coordinate system:
      #     X+: right, Y+: up, Z+: front.
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
