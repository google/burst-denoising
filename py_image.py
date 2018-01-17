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
"""Python image utilities.
"""

import numpy as np
import scipy as sp
import os
import time
from matplotlib import pyplot as plt
from tensorflow import gfile

# Basic utils

def p2s(p):
  return 10.**(-.05*p)

def s2p(s):
  return -20. * np.log10(s)

def awgn_like(imgs):
  return np.random.normal(size=imgs.shape)

def down2(im, batched=False):
  if not batched:
    im = im[np.newaxis, ...]
  im = im[:, :im.shape[1]//2*2, :im.shape[2]//2*2, ...]
  im = .25 * (im[:, ::2,::2,...]+
              im[:, 1::2,::2,...]+
              im[:, ::2,1::2,...]+
              im[:, 1::2,1::2,...])
  if not batched:
    im = im[0,...]
  return im

def downN(im, n, batched=False):
  for i in range(n):
    im = down2(im, batched)
  return im

def blur(imgs, f):
  N = f.shape[-1]
  sh = imgs.shape
  imgs = imgs.reshape([-1] + list(imgs.shape[-2:]))
  pd = N//2
  imgs = np.pad(imgs, [[0,0],[pd,0],[pd,0]], mode='constant')
  f = f.reshape([-1,N,N])
  imgs = sp.ndimage.filters.convolve(imgs, f)
  imgs = imgs[:, :-pd, :-pd]
  imgs = imgs.reshape(sh)
  return imgs


def gaussian_blur(imgs, sigma, N=9):
  t = np.linspace(-.5*(N-1), -5*(N-1), N)
  y, x = np.meshgrid(t, t)
  f = np.exp(-(x**2+y**2)/(2.*sigma**2))
  f = f / np.sum(f)
  return blur(imgs, f)

def box_blur(imgs, N=9):
  f = np.ones([N,N])
  f = f / np.sum(f)
  return blur(imgs, f)


def psnr(estimate, truth, bd=0, batched=False):
  sqdiff = np.square(estimate - truth)
  if not batched:
    sqdiff = sqdiff[np.newaxis,...]
  if bd != 0:
    sqdiff = sqdiff[:, bd:-bd, bd:-bd, ...]
  sqdiff = np.reshape(sqdiff, [sqdiff.shape[0], -1])
  eps = 1e-10
  val = -10. * (np.log10(np.maximum(1e-10, np.mean(sqdiff, axis=1))))
  return val if batched else np.mean(val)

def imshow(im, clip=True, gamma=1., nrmlz=1., cb=False, fig=False, figsize=None, v=None, title=None, no_axis=False):
  if figsize is not None:
    plt.figure(figsize=figsize)
  elif fig:
    plt.figure()
  im2show = im + 0
  im2show /= nrmlz
  if v is None:
    vmin = np.min(im2show)
    vmax = np.max(im2show)
  else:
    vmin, vmax = v
  im2show = np.squeeze(im2show)
  if clip:
    im2show = np.clip(im2show, 0, 1)
  im2show = im2show**gamma
  plt.imshow(im2show, interpolation='none', vmin=vmin, vmax=vmax)
  if title is not None:
    plt.title(title)
  if cb:
    plt.colorbar()
  if no_axis:
    plt.axis('off')

def imshow_row(ims, gamma=1., nrmlz=1., cb=False, fig=False, figsize=None, v=None, no_axis=False):
  if figsize is not None:
    plt.figure(figsize=figsize)
  elif fig:
    plt.figure()
  ims = [im/nrmlz for im in ims]
  for i, im in enumerate(ims):
    plt.subplot(1,len(ims),i+1)
    if isinstance(im, list):
      imshow(im[0], gamma=gamma, cb=cb, title=im[1], v=v, no_axis=no_axis)
    else:
      imshow(im, gamma=gamma, cb=cb, v=v, no_axis=no_axis)

def show_filts(filts, hw):
  h, w = hw
  fsh = filts.shape
  filts = np.reshape(filts, [fsh[0], fsh[1], h, w])
  filts = np.pad(filts, [[1,1],[1,1],[0,0],[0,0]], mode='constant')
  filts = np.transpose(filts, [2,0,3,1])
  filts = np.reshape(filts, [(fsh[0]+2)*h, (fsh[1]+2)*w])
  return filts

def imshow_filts(filts, hw=None, cb=False, figsize=None):
  if figsize is not None:
    plt.figure(figsize=figsize)
  if filts.ndim==5:
    filts = filts[0,...]
  if hw is None:
    hw = list(filts.shape[-2:])
    if hw[0] > 1 and hw[1] == 1:
      hw = [hw[1], hw[0]]
  plt.imshow(show_filts(filts, hw), interpolation='none', cmap='viridis')
  if cb:
    plt.colorbar()



def sRGBforward(x):
  b = .0031308
  gamma = 1./2.4
  # a = .055
  # k0 = 12.92
  a = 1./(1./(b**gamma*(1.-gamma))-1.)
  k0 = (1+a)*gamma*b**(gamma-1.)
  gammafn = lambda x : (1+a)*np.power(np.maximum(x,b),gamma)-a
  # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
  srgb = np.where(x < b, k0*x, gammafn(x))
  k1 = (1+a)*gamma
  srgb = np.where(x > 1, k1*x-k1+1, srgb)
  return srgb



def _linreg(x, y):
  x_mu = np.mean(x)
  y_mu = np.mean(y)
  a = (np.mean(x*y) - x_mu*y_mu) / (np.mean(x*x) - x_mu**2)
  b = y_mu - a * x_mu
  return a, b

def estimate_shot_read(base, true, N=32, verbose=False):
  squid = (base-true)**2
  bins = np.linspace(0, np.median(true), N+1)
  bin_c = .5 * (bins[:-1] + bins[1:])
  y = []
  x = []
  for i in range(N):
    true_pts = true[(true >= bins[i]) * (true < bins[i+1])]
    squid_pts = squid[(true >= bins[i]) * (true < bins[i+1])]
    if len(squid_pts) > 0:
      x.append(bin_c[i])
      y.append(np.mean(squid_pts))
  x, y = np.array(x), np.array(y)
  a, b = _linreg(x, y)
  if verbose:
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, x * a + b)
    plt.show()
  return a, b

# Super res, demosaic reshape ops

def split_axis(x, axis, a=-1, b=-2):
  sh = list(x.shape)
  sh = sh[:axis] + [a, b] + (sh[axis+1:] if axis != -1 else [])
  print 'split', x.shape, '->', sh
  return np.reshape(x, sh)

def merge_axes(x, axis):
  sh = list(x.shape)
  sh = sh[:axis] + [-1] + (sh[axis+2:] if axis != -2 else [])
  print 'merge', x.shape, '->', sh
  return np.reshape(x, sh)

def combine_greens(bstack, axis=-1):
  r = np.take(bstack, 0, axis=axis)
  g1 = np.take(bstack, 1, axis=axis)
  g2 = np.take(bstack, 2, axis=axis)
  b = np.take(bstack, 3, axis=axis)
  return np.stack([r, .5*(g1+g1), b], axis=axis)

def remosaic(imgs, axis):
  sh = imgs.shape
  imgs = np.reshape(imgs, [np.prod(sh[:axis])] + list(sh[axis:axis+3]) + [np.prod(sh[axis+3:])])
  imgs = np.stack(
      [
          imgs[:,  ::2,  ::2, 0, :],
          imgs[:,  ::2, 1::2, 1, :],
          imgs[:, 1::2,  ::2, 1, :],
          imgs[:, 1::2, 1::2, 2, :],
      ], axis=3)
  sh_ = list(sh)
  sh_[axis+0] = sh_[axis+0] // 2
  sh_[axis+1] = sh_[axis+1] // 2
  sh_[axis+2] = 4
  imgs = np.reshape(imgs, sh_)
  return imgs


def fold_channels(x, n, axis=-3):
  sh0 = x.shape
  x = np.reshape(x, [np.prod(sh0[:axis])] + list(sh0[axis:axis+2]) + [n, n, -1])
  sh = x.shape
  x = np.reshape(x, [sh[0], sh[1], sh[2], n, n, sh[-1]])
  x = np.transpose(x, [0, 1, 3, 2, 4, 5])
  x = np.reshape(x, [sh[0], sh[1]*n, sh[2]*n, sh[-1]])
  sh = list(sh0[:axis]) + [sh0[axis]*n, sh0[axis+1]*n, -1] + \
          (list(sh0[axis+3:]) if axis != -3 else [])
  x = np.reshape(x, sh)
  if x.shape[-1] == 1:
    x = x[...,0]
  return x
  # ndims = x.ndim
  # sh = np.shape(x)
  # if ndims==5:
  #   x = np.reshape(x, [sh[0], sh[1], sh[2], n, n, sh[-1]])
  #   x = np.transpose(x, [0, 1, 3, 2, 4, 5])
  #   x = np.reshape(x, [sh[0], sh[1]*n, sh[2]*n, sh[-1]])
  # else:
  #   x = np.reshape(x, [sh[0], sh[1], sh[2], n, n])
  #   x = np.transpose(x, [0, 1, 3, 2, 4])
  #   x = np.reshape(x, [sh[0], sh[1]*n, sh[2]*n])
  # return x

def unfold_channels(x, n, axis=-2):

  sh0 = x.shape
  x = np.reshape(x, [np.prod(sh0[:axis]), sh0[axis]//n, n, sh0[axis+1]//n, n, -1])
  x = np.transpose(x, [0, 1, 3, 2, 4, 5])
  sh = x.shape
  x = np.reshape(x, [sh[0], sh[1], sh[2], n*n, -1])
  sh = list(sh0[:axis]) + [sh0[axis]//n, sh0[axis+1]//n, n*n] + \
          (list(sh0[axis+2:]) if axis != -2 else [])
  # print sh0, x.shape, sh
  x = np.reshape(x, sh)
  # if x.shape[-1] == 1:
  #   x = x[...,0]
  return x

  # ndims = x.ndim
  # sh = np.shape(x)
  # if ndims==4:
  #   x = np.reshape(x, [sh[0], sh[1]//n, n, sh[2]//n, n, sh[-1]])
  #   x = np.transpose(x, [0, 1, 3, 2, 4, 5])
  #   x = np.reshape(x, [sh[0], sh[1]//n, sh[2]//n, n*n, sh[-1]])
  # else:
  #   x = np.reshape(x, [sh[0], sh[1]//n, n, sh[2]//n, n])
  #   x = np.transpose(x, [0, 1, 3, 2, 4])
  #   x = np.reshape(x, [sh[0], sh[1]//n, sh[2]//n, n*n])
  # return x

# convolve linear system ops

def solve_convolve(noisy, truth, final_K, excl_edges=False):
  kpad = final_K//2
  ch = noisy.shape[-1]
  ch1 = truth.shape[-1]
  sh = noisy.shape
  h, w = sh[1], sh[2]
  img_stack = []
  noisy = np.pad(noisy, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]], mode='constant')
  for i in range(final_K):
    for j in range(final_K):
      img_stack.append(noisy[:, i:h+i, j:w+j, :])
  img_stack = np.stack(img_stack, axis=-2) # [batch, h, w, K**2, ch]
  if excl_edges:
    img_stack = img_stack[:, kpad:-kpad, kpad:-kpad, :]
    truth = truth[:, kpad:-kpad, kpad:-kpad]
    h = h - 2*kpad
    w = w - 2*kpad
  A = np.reshape(img_stack, [sh[0], h*w, final_K**2 * ch])
  b = np.reshape(truth, [sh[0], h*w, ch1])
  x = []
  for i in range(sh[0]):
    x.append(np.linalg.lstsq(A[i,...], b[i,...])[0])
  x = np.stack(x, axis=0)
  x = np.reshape(x, [sh[0], final_K, final_K, ch, ch1])
  return x


def convolve(img_stack, filts, final_K, final_W, spatial=True):
  noisy = img_stack
  initial_W = img_stack.shape[-1]
  kpad = final_K//2
  ch = noisy.shape[-1]
  ch1 = final_W
  sh = noisy.shape
  h, w = sh[1], sh[2]
  noisy = np.pad(noisy, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]], mode='constant')
  img_stack = []
  for i in range(final_K):
    for j in range(final_K):
      img_stack.append(noisy[:, i:h+i, j:w+j, :])
  img_stack = np.stack(img_stack, axis=-2) # [batch, h, w, K**2, ch]

  A = np.reshape(img_stack, [sh[0], h, w, final_K**2 * ch, 1])

  fsh = filts.shape
  x = np.reshape(filts, [fsh[0], fsh[1] if spatial else 1, fsh[2] if spatial else 1,
                         final_K ** 2 * initial_W, final_W])

  return np.sum(A * x, axis=-2)


def optimal_convolve(noisy, truth, final_K, conv_stack=None):
  t_dim = truth.ndim
  if noisy.ndim==3:
    noisy = noisy[np.newaxis, ...]
  if truth.ndim==2:
    truth = truth[..., np.newaxis]
  if truth.ndim==3:
    truth = truth[np.newaxis, ...]

  filts = solve_convolve(noisy, truth, final_K, True)
  fsh = filts.shape
  filts_ = filts[:,np.newaxis,np.newaxis,...] #tf.expand_dims(tf.expand_dims(filts, axis=1), axis=1)
  final_W = truth.shape[-1]
  if conv_stack is None:
    conv_stack = noisy
  assert noisy.shape[-1] == conv_stack.shape[-1]
  shift1 = convolve(conv_stack, filts_, final_K, final_W)

  if t_dim==2:
    shift1 = shift1[0, ..., 0]
  if t_dim==3:
    shift1 = shift1[0, ...]

  return shift1, filts

# Random crop ops

def crop_radial(img, y, x, sy, sx):
  return img[y-sy:y+sy, x-sx:x+sx,...]

def padA2shape(a, b):
  padding = np.maximum(0, (b - a.shape + 1) // 2)
  padarg = [[p, p] for p in padding]
  a = np.pad(a, padarg, mode='constant')
  return a

def random_crop(img, sh, axis=0):
  b = np.zeros_like(img.shape)
  b[axis:axis+len(sh)] = sh
  img = padA2shape(img, b)
  for i, s in enumerate(sh):
    ax = i + axis
    crop = np.random.randint(0, img.shape[ax] - s)
    img = np.take(img, range(crop, crop+s), axis=ax)
  return img

def random_crop_prep(img, sh, axis=0):
  b = np.zeros_like(img.shape)
  b[axis:axis+len(shape)] = sh
  img = padA2shape(img, b)
  return img

def random_crop_cvals(img, sh, axis=0):
  c_vals = []
  for i, s in enumerate(sh):
    ax = i + axis
    crop = np.random.randint(0, img.shape[ax] - s)
    c_vals.append(crop)
  return c_vals

def random_crop_apply(img, c_vals, sh, axis=0):
  for i, s in enumerate(sh):
    ax = i + axis
    crop = c_vals[i]
    img = np.take(img, range(crop, crop+s), axis=ax)
  return img

def random_crop_unsafe(img, sh, axis=0):
  c_vals = []
  for i, s in enumerate(sh):
    ax = i + axis
    crop = np.random.randint(0, img.shape[ax] - s)
    c_vals.append(crop)
    img = np.take(img, range(crop, crop+s), axis=ax)
  return img

def random_patch_burst(img, sh, jit=1, factor=1, burst=2, cvals=None):
  pad = jit * 2**factor
  pad_size = [(s + 2*jit) * 2**factor for s in sh]
  reg_size = [s * 2**factor for s in sh]
  full_crop = np.all(np.array(img.shape) == np.array(pad_size))
  patch = img if full_crop else random_crop(img, pad_size)

  if cvals is None:
    cvals = [[pad,pad]]
    for b in range(burst-1):
      if jit > 0:
        cvals.append(random_crop_cvals(patch, reg_size))
      else:
        cvals.append(cvals[0])

  patches = [random_crop_apply(patch, cvals[0], reg_size)]
  for b in range(burst-1):
    patches.append(random_crop_apply(patch, cvals[b+1], reg_size))
  patches = np.stack(patches, axis=-1)
  patches = downN(patches, factor, batched=False)
  return patches, cvals

  # patches = [downN(patch[pad:-pad, pad:-pad, ...], factor)]
  # for b in range(burst-1):
  #   if jit > 0:
  #     patches.append(downN(random_crop_unsafe(patch, reg_size), factor))
  #   else:
  #     patches.append(patch[0])
  # patches = np.stack(patches, axis=-1)
  # return patches

def random_patches(img, sh, jit=1, factor=1, burst=2, batch=0, cv_repeat=1):
  if batch==0:
    return random_patch_burst(img, sh, jit, factor, burst)[0]
  else:
    patches = []
    for b in range(batch):
      if b % cv_repeat == 0:
        cvals = None
      patch, cvals = random_patch_burst(img, sh, jit, factor, burst, cvals)
      patches.append(patch)
    patches = np.stack(patches, axis=0)
    return patches

# HDR Plus

def rcwindow(N):
  x = np.linspace(0., N, N, endpoint=False)
  rcw = .5 - .5 * np.cos(2.*np.pi * (x + .5) / N)
  rcw = rcw.reshape((N,1)) * rcw.reshape((1,N))
  return rcw

def hdrplus_merge(imgs, c, sig, spatial=True):
  # imgs is [..., h, w, ch]
  rcw = rcwindow(imgs.shape[-2])[...,np.newaxis]
  imgs = imgs * rcw
  imgs_f = np.fft.fft2(imgs, axes=(-3,-2))
  Dz2 = np.square(np.abs(imgs_f[...,0:1] - imgs_f))
  Az = Dz2 / (Dz2 + c*sig**2)
  filts = 1 - Az
  filts[...,0] = 1 + np.sum(Az[...,1:], axis=-1)
  output_f = np.mean(imgs_f * filts, axis=-1)
  output_f = np.real(np.fft.ifft2(output_f))

  if spatial:

    output_stack = []
    filts_s = np.real(np.fft.ifft2(filts, axes=(-3,-2)))
    N = imgs.shape[-1]
    for i in range(N):
        in1 = imgs[...,i]
        in2 = filts_s[...,i]
        output_stack.append(np.fft.fftshift(sp.signal.convolve2d(in1, in2, mode='same', boundary='wrap')))
    output_stack = np.stack(output_stack, axis=-1)
    output_stack = np.roll(np.roll(output_stack,-1,axis=0),-1,axis=1)
    output_s = np.mean(output_stack, axis=-1)
    return imgs, output_f, output_s, filts, filts_s, Az, output_stack

  else:
    return imgs, output_f, filts, Az


def hdrplus_tiled(noisy, N, sig, c=10**2.5):
  sh = noisy.shape[0:2]
  buffer = np.zeros_like(noisy[...,0])
  for i in range(2):
    for j in range(2):
      nrolled = np.roll(np.roll(noisy, axis=0, shift=-N//2*i), axis=1, shift=-N//2*j)
      hpatches = (np.transpose(np.reshape(nrolled, [sh[0]//N, N, sh[1]//N, N, -1]), [0,2,1,3,4]))
      merged = hdrplus_merge(hpatches, c, sig, spatial=False)[1]
      merged = (np.reshape(np.transpose(merged, [0,2,1,3]), sh))
      merged = np.roll(np.roll(merged, axis=0, shift=N//2*i), axis=1, shift=N//2*j)
      buffer += merged
  return buffer


def hdrplus_tiled_sigbatch(noisy, N, sig, c=10**2.5):
  sh = noisy.shape[0:3]

  chunk = 16
  if sh[0] > chunk:
    buffer = []
    print 'tiling the', sh[0], 'last axis'
    for i in range(0, sh[0], chunk):
      print i,
      buffer.append(hdrplus_tiled_sigbatch(noisy[i:i+chunk, ...], N, sig[i:i+chunk,...], c))
    print 'done'
    return np.concatenate(buffer, axis=0)

  buffer = np.zeros_like(noisy[...,0])
  noisy_ = noisy
  noisy = np.concatenate([noisy, sig], axis=-1)
  for i in range(2):
    for j in range(2):
      nrolled = np.roll(np.roll(noisy, axis=1, shift=-N//2*i), axis=2, shift=-N//2*j)
      hpatches = (np.transpose(np.reshape(nrolled, [sh[0], sh[1]//N, N, sh[2]//N, N, -1]), [0,1,3,2,4,5]))

      patches = hpatches[...,:-1]
      sigpatches = hpatches[...,-1]
      sigpatches = np.sqrt(np.mean(np.mean(sigpatches**2, axis=-1), axis=-1))
      sigpatches = np.tile(np.reshape(sigpatches, list(sigpatches.shape) + [1,1,1]), [1,1,1,N,N,noisy_.shape[-1]])

      merged = hdrplus_merge(patches, c, sigpatches, spatial=False)[1]
      merged = (np.reshape(np.transpose(merged, [0,1,3,2,4]), sh))
      merged = np.roll(np.roll(merged, axis=1, shift=N//2*i), axis=2, shift=N//2*j)
      buffer += merged
  return buffer


def hdrplus_csearch(noisy, truth, N, sig, post_fn=None):
  c_central = 0.
  c_ranges = [np.linspace(-10,10,25), np.linspace(-1,1,25)]
  pvals = []
  for i, c_range in enumerate(c_ranges):
    recons = [hdrplus_tiled(noisy, N, sig, c=10**c) for c in c_central + c_range]
    if post_fn is not None:
      recons = map(post_fn, recons)
    psnrs = [psnr(r, truth) for r in recons]
    pvals.append([c_central + c_range, psnrs])
    c_central = c_central + c_range[np.argmax(psnrs)]
  return c_central, pvals

# Alignment


def small_align(img0, img1, y, x, ys, xs, bd, dy=0, dx=0):
  vals = []
  indices = []
  tile0 = img0[y:y+ys, x:x+xs]
  for i in range(y-bd, y+bd+1):
    for j in range(x-bd, x+bd+1):
      tile1 = img1[i+dy:i+dy+ys, j+dx:j+dx+xs]
      vals.append(np.mean((tile0-tile1)**2))
      indices.append([i,j])
  vals = np.array(vals)
  ind = np.argmin(vals)
  ind = indices[ind]
  vals = vals.reshape([2*bd+1,2*bd+1])
  tile1 = img1[ind[0]:ind[0]+ys, ind[1]:ind[1]+xs]
  return tile1, ind, vals, tile0


def roll2(x, i, j):
  return np.roll(np.roll(x, i, 0), j, 1)


def whole_img_align(img0, img1, bd, pd, verbose=False):
  img0 = np.mean(img0.reshape(list(img0.shape[:2])+[-1]), axis=-1)
  img1 = np.mean(img1.reshape(list(img1.shape[:2])+[-1]), axis=-1)
  vals = []
  indices = []
  tile0 = img0[pd:-pd, pd:-pd, ...]
  for i in range(-bd, bd+1):
    for j in range(-bd, bd+1):
      tile1 = roll2(img1, -i, -j)
      tile1 = tile1[pd:-pd, pd:-pd, ...]
      diff2 = (np.square(tile0-tile1))
      diff = np.mean(diff2)
      vals.append(diff)
      indices.append([bd+i,bd+j])
  vals = np.array(vals)
  ind = np.argmin(vals)
  ind = np.array(indices[ind]) - bd
  vals = vals.reshape([2*bd+1,2*bd+1])
  return ind, vals

def coarse2fine_align(img0, img1, bd, N=2):
  bd0 = bd // 2**N
  img1_ = img1 + 0.
  ind0, vals = whole_img_align(downN(img0, N), downN(img1, N), bd0, bd0)
  ind0 = np.array(ind0) * 2**N
  img1 = roll2(img1, -ind0[0], -ind0[1])
  ind, vals = whole_img_align(img0, img1, 2**N, bd)
  img1 = np.roll(np.roll(img1, -ind[0], 0), -ind[1], 1)
  pd = bd
  tile0 = img0[pd:-pd, pd:-pd, ...]
  tile1 = img1[pd:-pd, pd:-pd, ...]
  tile1_ = img1_[pd:-pd, pd:-pd, ...]
  print 'coarse {}, fine {}, net {}. gain of {} vs {}'.format(
      ind0, ind, ind0+ind,
      np.mean(np.square(tile1-tile0)),
      np.mean(np.square(tile1_-tile0)))
  return tile1, tile0

def mask_hot_pixels(im, thresh=.98):
  k = thresh
  mask = (im >= roll2(im,0,1)*k) * (im >= roll2(im,1,0)*k) * (im >= roll2(im,0,-1)*k)* (im >= roll2(im,-1,0)*k)
  mask = np.prod(mask, axis=-1)
  mask = np.tile(mask[...,np.newaxis], [1,1,im.shape[-1]])
  rep = .25*(roll2(im,0,1)+roll2(im,1,0)+roll2(im,0,-1)+roll2(im,-1,0))
  im[mask==1] = rep[mask==1]
  return im, mask[...,0]


def process_stack(raw_im_list, bls, raw_true=None, bl_true=None, bayer=False):
  # Average bayer blocks
  # Subtract black level
  if bayer:
    imsdn = np.stack([unfold_channels(im, 2, axis=0) for im in raw_im_list], axis=-1)
  else:
    imsdn = np.stack([im for im in raw_im_list], axis=-1)
  print 'Working with size {}'.format(imsdn.shape)
  print 'Average black level {}'.format(np.mean(bls))
  bls = bls.reshape([1,1,bls.shape[0],bls.shape[-1]]).transpose([0,1,3,2])
  if not bayer:
    bls = np.mean(bls, -2)
  imsdn = imsdn - bls

  # Repress hot pixels
  # mask = np.ones_like(imsdn[...,0])
  # k = .98 # threshold for hotness
  # for i in range(8):
  #   im = imsdn[...,i].astype(np.float64)
  #   m = (im >= roll2(im,0,1)*k) * (im >= roll2(im,1,0)*k) * (im >= roll2(im,0,-1)*k)* (im >= roll2(im,-1,0)*k)
  #   mask = mask * m

  # for i in range(8):
  #   im = imsdn[...,i]
  #   rep = .25*(roll2(im,0,1)+roll2(im,1,0)+roll2(im,0,-1)+roll2(im,-1,0))
  #   im[mask==1] = rep[mask==1]
  #   imsdn[...,i] = im

  imsdn, mask = mask_hot_pixels(imsdn)
  print 'Percent hot pixels = {:.4f}%'.format(100.*np.sum(mask==1)/mask.size)

  # Whole image align
  tiles = []
  bd = 16
  dumb = []
  for i in range(0,8):
    dumb.append(imsdn[bd:-bd,bd:-bd,...,i])
    tile1, tile0 = coarse2fine_align(imsdn[...,0], imsdn[...,i], bd=bd)
    tiles.append(tile1 if i > 0 else tile0)

### FIX below
  tiles = np.stack(tiles, axis=2)
  dumb = np.stack(dumb,axis=2)
  if bayer:
    tiles = tiles.transpose([3,0,1,2])
    dumb = dumb.transpose([3,0,1,2])
  tiles_score = np.mean(np.square(tiles - tiles[...,0:1]))
  dumb_score = np.mean(np.square(dumb - dumb[...,0:1]))

  print 'Alignment complete, total gain of {} vs {}'.format(tiles_score, dumb_score)

  if raw_true is None:
    return tiles, dumb

  # else:
  #   raw_true = downN(raw_true, 1)
  #   raw_true = raw_true - np.mean(bl_true)
  #   raw_true, _ = coarse2fine_align(imsdn[...,0], raw_true, bd=bd)
  #   return tiles, dumb, raw_true




def train_merge_simple_make_burst_test(raw_im_list, bls, sigs, log_dir, name, color=False):
  tiles, dumb = process_stack(raw_im_list, bls, bayer=color)
  print 'Tiles has shape {}'.format(tiles.shape)
  if not color:
    tiles = np.mean(tiles, axis=0)[np.newaxis, ...]
    dumb = np.mean(dumb, axis=0)[np.newaxis, ...]

  full_stack = tiles + 0.
  scale = 2. ** np.ceil(np.log2(np.max(np.array(raw_im_list))))
  full_stack = full_stack / scale
  print 'Scaling down by base max {} (max of tiles is {})'.format(scale, np.max(tiles))
  truth = np.mean(full_stack, axis=-1)
  noisy = full_stack
  sig_shot, sig_read = sigs[0,:,0], sigs[0,:,1]
  if not color:
    sig_shot = np.array([np.mean(sig_shot)]) / 2. # variance reduction from averaging!
    sig_read = np.array([np.mean(sig_read)]) / 2.
  print 'Taking shot and read as {} and {}'.format(sig_shot, sig_read)

  sr = sig_read.reshape([sig_read.shape[0], 1, 1, 1])
  ss = sig_shot.reshape([sig_shot.shape[0], 1, 1, 1])
  sig_read_single_std = np.sqrt(sr**2 + np.maximum(0., noisy[...,0:1]) * ss**2)

  full_batch = {
      'truth' : truth,
      'noisy' : noisy,
      'sig_read' : sig_read,
      'sig_shot' : sig_shot,
      'sig_read_single_std' : sig_read_single_std,
      'white_level' : np.array([1.] * noisy.shape[0])
  }

  batch, height, width, BURST_LENGTH = noisy.shape
  filename = 'burst.{}.{}x{}x{}x{}.npz'.format(name, batch, height, width, BURST_LENGTH)
  full_path = os.path.join(log_dir, filename)
  np.savez(full_path, **full_batch)
  print 'Saved {}'.format(filename)

  return tiles, dumb


# Eval utils





def tensor2patches(tensor, psize=512, bd=64):
  sh = tensor.shape
  nh = (sh[1]-1)//psize+1
  nw = (sh[2]-1)//psize+1

  if nh==1 and nw==1:
    patches = tensor

  else:
    patches = []
    tensor = tensor.reshape(list(sh[:3]) + [-1])
    tensor = np.pad(tensor, [[0,0],[bd,bd+psize],[bd,bd+psize],[0,0]], mode='constant')
    tensor = tensor.reshape(list(tensor.shape[:3]) + list(sh[3:]))
    for i in range(nh):
      for j in range(nw):
        y0 = bd + i * psize
        x0 = bd + j * psize
        patch = tensor[:, y0-bd:y0+bd+psize, x0-bd:x0+bd+psize, ...]
        patches.append(patch)
    patches = np.stack(patches, axis=1)
    patches = np.reshape(patches, [-1] + list(patches.shape[2:]))
    return patches

  return patches


def patches2tensor(patches, sh, psize=512, bd=64):
  psh = patches.shape
  nh = (sh[1]-1)//psize+1
  nw = (sh[2]-1)//psize+1

  if nw==1 and nh==1:
    tensor = patches
  else:
    patches = patches.reshape([-1, nh*nw] + list(psh[1:]))
    tensor = np.zeros([sh[0], psize*nh, psize*nw] + list(psh[3:]))
    for i in range(nh):
      for j in range(nw):
          y0 = i * psize
          x0 = j * psize
          tensor[:, y0:y0+psize, x0:x0+psize, ...] = patches[:, i*nw+j, bd:-bd, bd:-bd, ...]
    tensor = tensor[:, :sh[1], :sh[2], ...]

  return tensor



# def batchx4(batch_list):
#   new_batches = []
#   for batch in batch_list:
#     sh = batch.shape
#     batch = batch[:, :sh[1]//2*2, :sh[2]//2*2, ...]
#     new_batches.extend([
#         batch[:, :sh[1]//2, :sh[2]//2, ...],
#         batch[:, :sh[1]//2, sh[2]//2:, ...],
#         batch[:, sh[1]//2:, :sh[2]//2, ...],
#         batch[:, sh[1]//2:, sh[2]//2:, ...]
#     ])
#   return new_batches

def tt_new2old(train_tensor):
  base = train_tensor[...,0:1]
  sig_shot = train_tensor[...,9:10]
  sig_read = train_tensor[...,10:11]
  srss = np.sqrt(sig_read**2 + np.maximum(0.,base)*sig_shot**2)
  tt_old = np.concatenate([train_tensor[...,:9], srss], axis=-1)
  return tt_old


def testfiles2netinput(files, sigdup=0, old=False):
  print 'Loading files'
  dict_all = {}
  for i, f in enumerate(files):
    if i % (8 if len(files)>8 else 1) == 0:
      print i,
    example = np.load(gfile.Open(f))
    for d in example:
      if d not in dict_all:
        dict_all[d] = []
      dict_all[d].extend([example[d]])
  print i,'\n'

  print 'Prepping input'
  for d in dict_all:
    dict_all[d] = np.concatenate(dict_all[d], axis=0)
    print d, dict_all[d].shape

  dict_all['truth_all'] = dict_all['truth']
  if dict_all['truth'].ndim == 4:
    dict_all['truth'] = dict_all['truth'][...,0]

  if old:
    train_tensor = np.concatenate([
        dict_all['noisy'],
        dict_all['truth'][...,np.newaxis],
        dict_all['sig_read_single_std']
    ], axis=-1)
  else:
    sh = dict_all['noisy'].shape
    ss = dict_all['sig_shot'].reshape([sh[0],1,1,1])
    ss = np.tile(ss, [1, sh[1], sh[2], 1])
    sr = dict_all['sig_read'].reshape([sh[0],1,1,1])
    sr = np.tile(sr, [1, sh[1], sh[2], 1])
    train_tensor = np.concatenate([
        dict_all['noisy'],
        dict_all['truth'][...,np.newaxis],
        ss, sr
    ], axis=-1)


  k = 16
  train_tensor = train_tensor[:, :train_tensor.shape[1]//k*k, :train_tensor.shape[2]//k*k, ...]

  if sigdup > 0:
    tt = []
    for i in range(sigdup):
      base_t = 0.
      t = np.concatenate([train_tensor[...,:9],
               train_tensor[...,9:] * (base_t+(1-base_t)*(i)/float(sigdup-1))], axis=-1)
      tt.append(t)
    train_tensor = np.stack(tt, axis=1)
    train_tensor = train_tensor.reshape([-1] + list(train_tensor.shape[2:]))

  return train_tensor, dict_all


def invproc(noisy, wl=1., gamma=True, cliptop=1.):
  noisy_ = np.clip(np.transpose(np.transpose(noisy + 0.) / wl), 0., cliptop)
  return sRGBforward(noisy_) if gamma else noisy_

def quicksave(name, img, mode='bw'):
  if mode != 'bw':
    plt.imsave(name, img, cmap='viridis')
    return

  img = invproc(img, wl=1.)
  img = np.clip(img, 0, 1.)
  img = np.squeeze(img)
  plt.imsave(name, img, vmin=0, vmax=1, cmap='Greys_r')


def train_merge_simple_make_dslr_test(full_stack, log_dir, name, noise_params=None, scale=None):
  if scale is None:
    scale = 2**np.ceil(np.log2(np.max(full_stack[...,-1])))
  full_stack = full_stack / scale
  print 'Scaling down by truth max {}'.format(np.log2(scale))
  truth = full_stack[..., -1]
  noisy = full_stack[..., :-1]
  if noise_params is None:
    sig_shot, sig_read = estimate_shot_read(noisy[...,0], truth, N=64)
  else:
    sig_shot, sig_read = noise_params
  sig_read = np.maximum(0, sig_read)
  print 'Estimated shot and read as {} and {}'.format(sig_shot, sig_read)

  truth = truth[np.newaxis,...]
  noisy = noisy[np.newaxis,...]
  sig_read_single_std = np.sqrt(sig_read**2 + np.maximum(0., noisy[...,0:1]) * sig_shot**2)

  full_batch = {
      'truth' : truth,
      'noisy' : noisy,
      'sig_read' : np.array([sig_read]),
      'sig_shot' : np.array([sig_shot]),
      'sig_read_single_std' : sig_read_single_std,
      'white_level' : np.array([1.])
  }

  batch, height, width, BURST_LENGTH = noisy.shape
  filename = 'dslr_{}_{}x{}x{}x{}.npz'.format(name, batch, height, width, BURST_LENGTH)
  full_path = os.path.join(log_dir, filename)
  np.savez(full_path, **full_batch)


def explode(pic, N):
  if pic.ndim==2:
    pic = np.reshape(pic, (pic.shape[0], 1, pic.shape[1], 1))
    pic = np.tile(pic, (1,N,1,N))
    pic = np.reshape(pic, (pic.shape[0]*N, pic.shape[2]*N))
  else:
    pic = np.reshape(pic, (pic.shape[0], 1, pic.shape[1], 1, -1))
    pic = np.tile(pic, (1,N,1,N,1))
    pic = np.reshape(pic, (pic.shape[0]*N, pic.shape[2]*N, -1))
  return pic

def gen_html_page(pics_list,
                  log_dir, # which experiment
                  out_file='comparisons', # where to put page
                  explode_factor=1, # how much to magnify
                  localdir=time.strftime("%m_%d_%y"), # subdir for pics
                  basedir='',
                  templ_file='',
                  header_freq=0, constrain=False):

  prefix = out_file[:out_file.find('.')]
  localdir = '{}_{}'.format(out_file,localdir)
  outdir = os.path.join(basedir, localdir)
  gfile.MakeDirs(outdir)
  print outdir

  out_file += '.html'

  with open(templ_file) as tf:
    header = tf.read()


  out_file = os.path.join(basedir, out_file)
  with open(out_file, 'w') as outf:
    outf.write(header)
    outf.write('\n <body>')
    name = 'Results test'
    details = 'Results on ' + log_dir
    outf.write('<h1>' + name + '</h1>')
    outf.write('<p>' + details + '</p>')

    """
    pics_list = [
        [
            key,
            [
                [pic_string, pic],
                [pic_string, pic],
                ...
            ]
        ],
        ...
    ]
    """

    outf.write('<table>')

    table_header = ''
    table_header += '<tr>\n'
    pkeys = [p[0] for p in pics_list]
    for pp in pkeys:
      table_header += '<th>' + pp + '</th>\n'
    table_header += '</tr>\n'
    outf.write(table_header)

    for i in range(len(pics_list[0][1])):
      if header_freq > 0 and i > 0 and i%header_freq==0:
        outf.write(table_header)
      outf.write('<tr> \n'.format(i))
      for j in range(len(pkeys)):
#         print i, j, pics_list[j][0]
        picfile = prefix + '_' + pics_list[j][0] + '_' + str(i) + '.png'
        for ch in [':',' ','@','(',')']:
          picfile = picfile.replace(ch, '_')
        picname = os.path.join(outdir, picfile)
        plt.imsave(picname, explode(pics_list[j][1][i][1], explode_factor), vmin=0, vmax=1)
        pic_path = os.path.join(localdir, picfile)
        pic_link = '<a href="{0}"><img src="{0}" {1}></a>'.format(pic_path, 'width="400"' if constrain else "")
        outf.write('<td> {} <br> [{}] {} </td>\n'.format(pic_link, i, pics_list[j][1][i][0]))

      outf.write('</tr>\n')

    outf.write('</body> </html>')
    print 'Done html, saved to {}'.format(out_file)
