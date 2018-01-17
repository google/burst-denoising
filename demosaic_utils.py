# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Demosaic utilities.

This is an old version of tf_image.py. Some demosaic related stuff hasn't yet
been migrated.
"""

import tensorflow as tf

def small_bayer_stack(bayer):
  r = bayer[:,::2,::2, :]
  g1 = bayer[:,1::2,::2, :]
  g2 = bayer[:,::2,1::2, :]
  b = bayer[:,1::2,1::2, :]
#   s = tf.tile(sigma, [1, tf.shape(r)[1], tf.shape(r)[2]])
  stack = tf.concat((r, g1, g2, b), axis=-1)
  return stack

def small_bayer_stack4(bayer):
  r = bayer[:,::2,::2, :]
  g1 = bayer[:,1::2,::2, :]
  g2 = bayer[:,::2,1::2, :]
  b = bayer[:,1::2,1::2, :]
#   s = tf.tile(sigma, [1, tf.shape(r)[1], tf.shape(r)[2]])
  stack = tf.stack((r, g1, g2, b), axis=-1)
  stack = tf.transpose(stack, [0, 1, 2, 4, 3])
  return stack

def tf22reshape(t):
  sh = tf.shape(t)
  t = tf.reshape(t, (sh[0], sh[1], sh[2], 2, 2))
  t = tf.transpose(t, (0, 1, 3, 2, 4))
  t = tf.reshape(t, (sh[0], sh[1]*2, sh[2]*2))
  return t

def tf22reshape1(t):
  sh = tf.shape(t)
  t = tf.reshape(t, (sh[0], sh[1], sh[2], 2, 2))
  t = tf.transpose(t, (0, 1, 3, 2, 4))
  t = tf.reshape(t, (sh[0], sh[1]*2, sh[2]*2, 1))
  return t

def tf22reshape2(t, BURST_LENGTH):
  sh = tf.shape(t)
  t = tf.reshape(t, (sh[0], sh[1], sh[2], BURST_LENGTH, 2, 2))
  t = tf.transpose(t, (0, 1, 4, 2, 5, 3))
  t = tf.reshape(t, (sh[0], sh[1]*2, sh[2]*2, BURST_LENGTH))
  return t

def full_bayer_stack(bayer, net, BURST_LENGTH):
  r = bayer[:,::2,::2, :]
  g1 = bayer[:,1::2,::2, :]
  g2 = bayer[:,::2,1::2, :]
  b = bayer[:,1::2,1::2, :]
  R  = tf22reshape2(tf.pad(tf.expand_dims(r, axis=-1), [[0,0],[0,0],[0,0],[0,0],[0,3]]), BURST_LENGTH)
  G1 = tf22reshape2(tf.pad(tf.expand_dims(g1, axis=-1), [[0,0],[0,0],[0,0],[0,0],[1,2]]), BURST_LENGTH)
  G2 = tf22reshape2(tf.pad(tf.expand_dims(g2, axis=-1), [[0,0],[0,0],[0,0],[0,0],[2,1]]), BURST_LENGTH)
  B  = tf22reshape2(tf.pad(tf.expand_dims(b, axis=-1), [[0,0],[0,0],[0,0],[0,0],[3,0]]), BURST_LENGTH)
  G = G1 + G2
  Rnet = tf22reshape1(net[...,:4])
  Gnet = tf22reshape1(net[...,4:8])
  Bnet = tf22reshape1(net[...,8:])
  stack = tf.concat((R,G,B,Rnet,Gnet,Bnet), axis=-1)
  return stack

def full_stack(imgs, net, BURST_LENGTH):
  r = imgs[...,:BURST_LENGTH]
  g1 = imgs[...,BURST_LENGTH:2*BURST_LENGTH]
  g2 = imgs[...,2*BURST_LENGTH:3*BURST_LENGTH]
  b = imgs[...,3*BURST_LENGTH:]
  R  = tf22reshape2(tf.pad(tf.expand_dims(r, axis=-1), [[0,0],[0,0],[0,0],[0,0],[0,3]]), BURST_LENGTH)
  G1 = tf22reshape2(tf.pad(tf.expand_dims(g1, axis=-1), [[0,0],[0,0],[0,0],[0,0],[1,2]]), BURST_LENGTH)
  G2 = tf22reshape2(tf.pad(tf.expand_dims(g2, axis=-1), [[0,0],[0,0],[0,0],[0,0],[2,1]]), BURST_LENGTH)
  B  = tf22reshape2(tf.pad(tf.expand_dims(b, axis=-1), [[0,0],[0,0],[0,0],[0,0],[3,0]]), BURST_LENGTH)
  G = G1 + G2
  Rnet = tf22reshape1(net[...,:4])
  Gnet = tf22reshape1(net[...,4:8])
  Bnet = tf22reshape1(net[...,8:])
  stack = tf.concat((R,G,B,Rnet,Gnet,Bnet), axis=-1)
  return stack

#####################################

def batch_down2(img):
  return (img[:,::2,::2,...]+img[:,1::2,::2,...]+img[:,::2,1::2,...]+img[:,1::2,1::2,...])/4

def batch_down2rgb(img):
  return tf.stack([img[:,::2,::2,...],(img[:,1::2,::2,...]+img[:,::2,1::2,...])/2,img[:,1::2,1::2,...]],axis=-1)

def down2(img):
  return (img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])/4

def gather2dgrid(image, grid_x, grid_y, grid_z):
  return tf.gather_nd(image, tf.stack((grid_z,grid_y,grid_x), axis=-1))

def bilerp(image, grid_x, grid_y, grid_z):
  grid_x0 = tf.cast(grid_x, tf.int32)
  grid_x1 = tf.cast(grid_x, tf.int32) + 1
  grid_y0 = tf.cast(grid_y, tf.int32)
  grid_y1 = tf.cast(grid_y, tf.int32) + 1
  grid_x0f = tf.cast(grid_x0, tf.float32)
  grid_x1f = tf.cast(grid_x1, tf.float32)
  grid_y0f = tf.cast(grid_y0, tf.float32)
  grid_y1f = tf.cast(grid_y1, tf.float32)

  t_00 = tf.tile(tf.expand_dims((grid_x1f - grid_x) * (grid_y1f - grid_y), axis=-1), [1,1,1,3])
  t_01 = tf.tile(tf.expand_dims((grid_x - grid_x0f) * (grid_y1f - grid_y), axis=-1), [1,1,1,3])
  t_10 = tf.tile(tf.expand_dims((grid_x1f - grid_x) * (grid_y - grid_y0f), axis=-1), [1,1,1,3])
  t_11 = tf.tile(tf.expand_dims((grid_x - grid_x0f) * (grid_y - grid_y0f), axis=-1), [1,1,1,3])

  g_00 = gather2dgrid(image, grid_x0, grid_y0, grid_z)
  g_01 = gather2dgrid(image, grid_x1, grid_y0, grid_z)
  g_10 = gather2dgrid(image, grid_x0, grid_y1, grid_z)
  g_11 = gather2dgrid(image, grid_x1, grid_y1, grid_z)

  return g_00 * t_00 + g_01 * t_01 + g_10 * t_10 + g_11 * t_11

def flow_gather_tf(truth, flow):
  Z, Y, X = tf.meshgrid(tf.range(tf.shape(flow)[0]), tf.range(tf.shape(flow)[1]), tf.range(0, tf.shape(flow)[2]), indexing='ij')
  return bilerp(truth, tf.clip_by_value(flow[...,0] + tf.cast(X, tf.float32), 0, tf.cast(tf.shape(truth)[2],tf.float32)-1.0001),
                       tf.clip_by_value(flow[...,1] + tf.cast(Y, tf.float32), 0, tf.cast(tf.shape(truth)[1],tf.float32)-1.0001),
                       tf.cast(Z, tf.int32))

def flow_patches(truth, flow, patchsize):
  flow = tf.tile(tf.reshape(flow, [tf.shape(flow)[0], tf.shape(flow)[1], tf.shape(flow)[2], 1, 1, 2]), [1, 1, 1, patchsize, patchsize, 1])
  flow = tf.transpose(flow, [0, 1, 3, 2, 4, 5])
  flow = tf.reshape(flow, [tf.shape(flow)[0], tf.shape(flow)[1] * patchsize, tf.shape(flow)[3] * patchsize, 2])
  return flow_gather_tf(truth, flow)

def flow_patches_int(truth, flow, patchsize):
  flow = tf.tile(tf.reshape(flow, [tf.shape(flow)[0], tf.shape(flow)[1], tf.shape(flow)[2], 1, 1, 2]), [1, 1, 1, patchsize, patchsize, 1])
  flow = tf.transpose(flow, [0, 1, 3, 2, 4, 5])
  flow = tf.reshape(flow, [tf.shape(flow)[0], tf.shape(flow)[1] * patchsize, tf.shape(flow)[3] * patchsize, 2])
  Z, Y, X = tf.meshgrid(tf.range(tf.shape(flow)[0]), tf.range(tf.shape(flow)[1]), tf.range(0, tf.shape(flow)[2]), indexing='ij')
  Y = tf.clip_by_value(flow[...,1] + Y, 0, tf.shape(truth)[1]-1)
  X = tf.clip_by_value(flow[...,0] + X, 0, tf.shape(truth)[2]-1)
  return tf.gather_nd(truth, tf.stack([Z, Y, X], axis=-1))

#####################################

def sRGBforward(x):
  b = .0031308
  gamma = 1./2.4
  # a = .055
  # k0 = 12.92
  a = 1./(1./(b**gamma*(1.-gamma))-1.)
  k0 = (1+a)*gamma*b**(gamma-1.)
  gammafn = lambda x : (1+a)*tf.pow(tf.maximum(x,b),gamma)-a
  # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
  srgb = tf.where(x < b, k0*x, gammafn(x))
  k1 = (1+a)*gamma
  srgb = tf.where(x > 1, k1*x-k1+1, srgb)
  return srgb

def dumb_tf_demosaic(mosaic):
  mpad = tf.pad(mosaic,[[0,0],[2,2],[2,2]])
  r11 = mpad[:,2:-2:2,2:-2:2]
  g21 = mpad[:,3:-1:2,2:-2:2]
  g12 = mpad[:,2:-2:2,3:-1:2]
  b22 = mpad[:,3:-1:2,3:-1:2]
  g11 = (mpad[:,1:-3:2,2:-2:2]+mpad[:,3:-1:2,2:-2:2]+mpad[:,2:-2:2,1:-3:2]+mpad[:,2:-2:2,3:-1:2]) / 4
  b11 = (mpad[:,1:-3:2,1:-3:2]+mpad[:,3:-1:2,1:-3:2]+mpad[:,1:-3:2,3:-1:2]+mpad[:,3:-1:2,3:-1:2]) / 4
  g22 = (mpad[:,2:-2:2,3:-1:2]+mpad[:,4::2,  3:-1:2]+mpad[:,3:-1:2,2:-2:2]+mpad[:,3:-1:2,4::2  ]) / 4
  r22 = (mpad[:,2:-2:2,2:-2:2]+mpad[:,4::2,  2:-2:2]+mpad[:,2:-2:2,4::2  ]+mpad[:,4::2,  4::2  ]) / 4
  r21 = (mpad[:,2:-2:2,2:-2:2]+mpad[:,4::2,  2:-2:2]) / 2
  b21 = (mpad[:,3:-1:2,1:-3:2]+mpad[:,3:-1:2,3:-1:2]) / 2
  r12 = (mpad[:,2:-2:2,2:-2:2]+mpad[:,2:-2:2,4::2  ]) / 2
  b12 = (mpad[:,1:-3:2,3:-1:2]+mpad[:,3:-1:2,3:-1:2]) / 2
  R = tf.stack((r11,r12,r21,r22),axis=-1)
  G = tf.stack((g11,g12,g21,g22),axis=-1)
  B = tf.stack((b11,b12,b21,b22),axis=-1)
  return tf.stack((tf22reshape(R),tf22reshape(G),tf22reshape(B)),axis=-1)


def process_for_tboard(ims, gamma=1./2.2):
  if ims.get_shape().ndims == 3:
    ims = tf.expand_dims(ims, axis=-1)
  return tf.cast((tf.clip_by_value(ims, 0.,1.)**gamma * 255), tf.uint8)

def decimatergb(rgb,bl):
  r = rgb[:,::2,::2,...,0]
  g1 = rgb[:,1::2,::2,...,1]
  g2 = rgb[:,::2,1::2,...,1]
  b = rgb[:,1::2,1::2,...,2]
  return tf22reshape2(tf.stack([r,g1,g2,b],axis=-1), bl)

def batch_down2(img):
  return (img[:,::2,::2,...]+img[:,1::2,::2,...]+img[:,::2,1::2,...]+img[:,1::2,1::2,...])/4

def twelveTo3(rgb):
  sh = tf.shape(rgb)
  return tf.reshape(tf.transpose(tf.reshape(rgb, [sh[0], sh[1], sh[2], 2, 2, 3]), [0,1,3,2,4,5]), [sh[0], sh[1]*2, sh[2]*2, 3])

def unstack_bayer(b):
  sh = tf.shape(b)
  return tf.transpose(tf.reshape(b, [sh[0],sh[1]/2,2,sh[2]/2,2]),[0,1,3,2,4])

def stack_bayer(b):
  sh = tf.shape(b)
  return tf.reshape(tf.transpose(tf.reshape(b,[sh[0],sh[1],sh[2],2,2]),[0,1,3,2,4]),[sh[0],sh[1]*2,sh[2]*2])

def psnr_tf(estimate, truth):
  return -10. * tf.log(tf.reduce_mean(tf.square(estimate - truth))) / tf.log(10.)


def psnr_tf_batch(estimate, truth):
  return tf.reduce_mean(-10. * tf.log(tf.reduce_mean(tf.reshape(tf.square(estimate - truth), [tf.shape(estimate)[0],-1]), axis=1)) / tf.log(10.))


def psnr_tf_batch_col(estimate, truth):
  return tf.reduce_mean(-10. * tf.log(tf.reduce_mean(tf.square(estimate - truth), axis=[1,2,3])) / tf.log(10.))


def psnr_tf_batch_col_safe(estimate, truth):
  msq = tf.reduce_mean(tf.square(estimate - truth), axis=[1,2,3])
  denom = tf.reduce_sum(tf.where(msq == 0., tf.zeros_like(msq), tf.ones_like(msq)))
  denom = tf.cond(denom < 1., lambda: 1., lambda: denom)
  msq = tf.where(msq == 0., tf.ones_like(msq), msq)
  return tf.reduce_sum(-10. * tf.log(msq) / tf.log(10.)) / denom


def add_read_shot_tf(truth, sig_read, sig_shot):
  read = sig_read * tf.random_normal(tf.shape(truth))
  shot = tf.sqrt(truth) * sig_shot * tf.random_normal(tf.shape(truth))
  noisy = truth + shot + read
  return noisy, batch_down2(tf.sqrt(noisy * sig_shot ** 2 + sig_read ** 2))

def add_read_shot22_tf(truth, sig_read, sig_shot):
  sh = tf.shape(truth)
  truth = tf.reshape(truth, [sh[0], sh[1]/2, 2, sh[2]/2, 2, sh[3]])
  sig_read = tf.reshape(sig_read, [-1, 1, 2, 1, 2, 1])
  sig_shot = tf.reshape(sig_shot, [-1, 1, 2, 1, 2, 1])
  shot = sig_shot * tf.random_normal(tf.shape(truth))
  read = tf.sqrt(truth) * sig_read * tf.random_normal(tf.shape(truth))
  noisy = truth + shot + read
  noisy = tf.reshape(truth, sh)
  return noisy


def atan2(x, y, epsilon=1.0e-12):
  # Add a small number to all zeros, to avoid division by zero:
  x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
  y = tf.where(tf.equal(y, 0.0), y+epsilon, y)

  pi = 3.1415926535
  angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
  angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + pi, angle)
  angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - pi, angle)
  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*pi * tf.ones_like(x), angle)
  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*pi * tf.ones_like(x), angle)
  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
  return angle


def vector2hsv_tf(flow):
  radius = tf.sqrt(tf.square(flow[...,0])+tf.square(flow[...,1]))
  radius = radius / tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_max(radius, axis=[1,2]), axis=1), axis=1), (1,tf.shape(radius)[1],tf.shape(radius)[2]))
  theta = atan2(flow[...,0], flow[...,1]) / (2. * 3.1416) + .5
  hsv_image = tf.stack([theta,radius,tf.ones_like(theta)], axis=-1)
  return tf.image.hsv_to_rgb(hsv_image)

def pullaway_loss(batch):
  sh = tf.shape(batch)
  b = sh[0]
  b0 = tf.tile(tf.reshape(batch, [sh[0],1,sh[1],sh[2],sh[3]]), [1, b, 1, 1, 1])
  b1 = tf.tile(tf.reshape(batch, [1,sh[0],sh[1],sh[2],sh[3]]), [b, 1, 1, 1, 1])
#   b0 = tf.reshape(b0, [b*b, sh[1],sh[2],sh[3]])
#   b1 = tf.reshape(b1, [b*b, sh[1],sh[2],sh[3]])
  numer = tf.square(tf.reduce_sum(b0*b1, axis=-1))
  denom = tf.reduce_sum(b0*b0, axis=-1) * tf.reduce_sum(b1*b1, axis=-1)
  denom = tf.where(tf.equal(denom,0.0), tf.zeros_like(denom), denom)
  dists = numer / denom
  dists = tf.reduce_mean(dists, axis=[2,3])
  x,y = tf.meshgrid(tf.range(tf.shape(dists)[0]), tf.range(tf.shape(dists)[1]), indexing='ij')
  dists = tf.where(tf.equal(x, y), tf.zeros_like(dists), dists)
  return tf.reduce_sum(dists) / tf.cast(b * (b-1), tf.float32)

