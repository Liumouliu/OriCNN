import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def constant_variable(var_shape, value=0.0, name='biases'):
  return tf.get_variable(name=name, shape=var_shape, 
      initializer=tf.constant_initializer(value=value))

def trace_locations_forward(loc_x, loc_y, im_size, feat_size):
  '''
  loc_x/y: pixel location in input image
  im_size: size of input image
  feat_size: size of feature maps
  '''
  feat_w_float = tf.cast(feat_size[1], tf.float32)
  feat_h_float = tf.cast(feat_size[0], tf.float32)

  # estimate the accumulated paddings
  poolx = tf.cast(im_size[1], tf.float32) / feat_w_float
  pooly = tf.cast(im_size[0], tf.float32) / feat_h_float
  padx = (poolx-1.) / 2.
  pady = (pooly-1.) / 2.

  # compute corresponding locations in feat map (paddings are considered)
  loc_x_feat = (tf.cast(loc_x,tf.float32) - padx) / poolx
  loc_y_feat = (tf.cast(loc_y,tf.float32) - pady) / pooly

  # scale the location values from [0, width/height - 1] to [-1, 1]
  loc_x_norm = loc_x_feat * 2. / (feat_w_float-1.) - 1.
  loc_y_norm = loc_y_feat * 2. / (feat_h_float-1.) - 1.
  location = tf.stack([loc_x_norm, loc_y_norm], axis=2)

  return location

def trace_locations_backward(loc_x, loc_y, anchor_size, feat_size):
  '''
  loc_x/y: pixel location in last feature map
  im_size: size of the smallest feature map
  feat_size: size of feature maps
  '''
  feat_w_float = tf.cast(feat_size[1], tf.float32)
  feat_h_float = tf.cast(feat_size[0], tf.float32)

  # estimate the accumulated paddings
  poolx = feat_w_float / tf.cast(anchor_size[1], tf.float32)
  pooly = feat_h_float / tf.cast(anchor_size[0], tf.float32)
  padx = (poolx-1.) / 2.
  pady = (pooly-1.) / 2.

  # compute corresponding locations in feat map (paddings are considered)
  loc_x_feat = tf.cast(loc_x,tf.float32) * poolx + padx
  loc_y_feat = tf.cast(loc_y,tf.float32) * pooly + pady

  # scale the location values from [0, width/height - 1] to [-1, 1]
  loc_x_norm = loc_x_feat * 2. / (feat_w_float-1.) - 1.
  loc_y_norm = loc_y_feat * 2. / (feat_h_float-1.) - 1.
  location = tf.stack([loc_x_norm, loc_y_norm], axis=2)

  return location

def extract_value(z, locs):
  '''
  z : b x h x w x c tensor 
  locs : b x n x 2 tensor
  '''

  X = tf.slice(locs, [0,0,0], [-1,-1,1])
  Y = tf.slice(locs, [0,0,1], [-1,-1,1])

  out = interpolate(z, X, Y)

  b = tf.shape(z)[0]
  c = tf.shape(z)[3]
  # b,_,_,c = z.get_shape().as_list()


  return tf.reshape(out, [b, -1, 1, c])

def extract_values(zs, locs):

  return tf.concat([extract_value(z, loc) for z, loc in zip(zs, locs)], 3)


def _repeat(x, n_repeats):
  with tf.variable_scope('_repeat'):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

def interpolate(im, X, Y):
  with tf.variable_scope('interpolate'):
    # constants
    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    # flatten X, Y (with size b x n x 1)
    out_height = tf.shape(X)[1]
    out_width = tf.shape(X)[2]
    x_flat = tf.reshape(X, [-1])
    y_flat = tf.reshape(Y, [-1])

    x = tf.cast(x_flat, 'float32')
    y = tf.cast(y_flat, 'float32')
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.zeros([], dtype='int32')
    max_y = tf.cast(height - 1, 'int32')
    max_x = tf.cast(width - 1, 'int32')

    # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0)*(width_f-1) / 2.0
    y = (y + 1.0)*(height_f-1) / 2.0

    # do sampling
    x0_f = tf.floor(x); x1_f = x0_f + 1.
    y0_f = tf.floor(y); y1_f = y0_f + 1.

    x0 = tf.cast(x0_f, 'int32')
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.cast(x1_f, 'int32')
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.cast(y0_f, 'int32')
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.cast(y1_f, 'int32')
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim2 = width
    dim1 = width*height
    base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.cast(im_flat, 'float32')
    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    wa = tf.expand_dims(((1-x+x0_f) * (1-y+y0_f)), 1)
    wb = tf.expand_dims(((1-x+x0_f) * (1-y1_f+y)), 1)
    wc = tf.expand_dims(((1-x1_f+x) * (1-y+y0_f)), 1)
    wd = tf.expand_dims(((1-x1_f+x) * (1-y1_f+y)), 1)

    output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return output
