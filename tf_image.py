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
"""Tensorflow image utilities.
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import py_image


# Helper for tensorflow training

def filts2imgs(filts, h, w):
  K = tf.shape(filts)[1]
  ch = tf.shape(filts)[3]
  filts = tf.reshape(filts, [-1, K, K, h, w])
  filts = tf.pad(filts, [[0,0],[1,1],[1,1],[0,0],[0,0]])
  filts = tf.transpose(filts, [0, 3, 1, 4, 2])
  filts = tf.reshape(filts, [-1, h*(K+2), w*(K+2), 1])
  return filts



def store_plot(plots, name, scalar, label=""):
  if name not in plots:
    plots[name] = []
  plots[name].append([label, scalar])

  return plots

def gen_plots(plots, g_index):
  summaries = []
  for name in plots:
    plot = plots[name]
    # plot.sort(key=lambda x : x[0])
    scalars = []
    i = 0
    for label, scalar in plot:
      scalars.append(scalar)
      name += '_' + str(i) + '_' + label
      i += 1
      tensor = tf.reshape(tf.stack(scalars), [len(scalars)])
    scalar = tf.cond(g_index < len(scalars), lambda: tensor[g_index], lambda: tensor[0])
    summaries.append(tf.summary.scalar(name, scalar))
    print 'Generating plot with name', name
  return tf.summary.merge(summaries)


def run_summaries(sess, fdict, writers, summaries, g_index, step):
  num_writers = len(writers)
  for i in range(num_writers):
    fdict[g_index] = i
    summaries_out, = sess.run([summaries], feed_dict=fdict)
    writers[i].add_summary(summaries_out, step)




# Basic

def batch_down2(img):
  return (img[:,::2,::2,...]+img[:,1::2,::2,...]+img[:,::2,1::2,...]+img[:,1::2,1::2,...])/4

def batch_down2rgb(img):
  return tf.stack([img[:,::2,::2,...],(img[:,1::2,::2,...]+img[:,::2,1::2,...])/2,img[:,1::2,1::2,...]],axis=-1)

def down2(img):
  return (img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])/4


# Loss



def gradient(imgs):
  return tf.stack([.5*(imgs[...,1:,:-1]-imgs[...,:-1,:-1]), .5*(imgs[...,:-1,1:]-imgs[...,:-1,:-1])], axis=-1)

def gradient_loss(guess, truth):
  return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))

def basic_img_loss(img, truth):
  l2_pixel = tf.reduce_mean(tf.square(img - truth))
  l1_grad = gradient_loss(img, truth)
  return l2_pixel + l1_grad


# SSIM


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value



# Eval stuff

def ckpt_num(ckpt):
  if 'model.ckpt-' not in ckpt:
    ckpt = tf.train.latest_checkpoint(ckpt)
  if ckpt is not None:
    ckpt = ckpt[ckpt.find('model.ckpt')+11:]
    ckpt = int(ckpt)
    return ckpt
  else:
    return -1


def print_keys_merge_simple(log_dir):
  g = tf.Graph()
  with g.as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

      ckpt_path = log_dir
      if 'model.ckpt' not in ckpt_path:
        ckpt_path = tf.train.latest_checkpoint(log_dir)

      if ckpt_path is not None:
        print 'Restoring from',ckpt_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        print 'Meta restored'
      else:
        print 'No checkpoint found in {}'.format(ckpt_path)
        return None

      var_col = tf.get_collection('inputs')
      noisy = var_col[0]
      dt = var_col[1]
      sig_read = var_col[2]
      output_ = tf.get_collection('output')
      output = []
      for out in output_:
        if 'dnet' in out.name:
          output.append(out)
      filters_ = tf.get_collection('filters')
      filters = []
      for f in filters_:
        filters.append(f)

      print 'output keys'
      for k in output:
        print k
      print 'filter keys'
      for k in filters:
        print k



# def test_merge_simple(log_dir, noisy_in, truth_in, sig_in):
#   g = tf.Graph()
#   with g.as_default():
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     with tf.Session(config=config) as sess:
#
#       ckpt_path = log_dir
#       if 'model.ckpt' not in ckpt_path:
#         ckpt_path = tf.train.latest_checkpoint(log_dir)
#
#       if ckpt_path is not None:
#         print 'Restoring from',ckpt_path
#         saver = tf.train.import_meta_graph(ckpt_path + '.meta')
#         print 'Meta restored'
#       else:
#         print 'No checkpoint found in {}'.format(ckpt_path)
#         return None
#
#       var_col = tf.get_collection('inputs')
#       noisy = var_col[0]
#       dt = var_col[1]
#       sig_read = var_col[2]
#       output_ = tf.get_collection('output')
#       output = []
#       for out in output_:
#         if 'dnet' in out.name:
#           output.append(out)
#       filters_ = tf.get_collection('filters')
#       filters = []
#       for f in filters_:
#         filters.append(f)
#
#       saver.restore(sess, ckpt_path)
#       print 'Weights restored'
#
#       def output2dict(out_tf, out_np):
#         ret = {}
#         for i in range(len(out_tf)):
#           ret[out_tf[i].name] = out_np[i]
#         return ret
#
#       def dict_combine(dict1, dict2):
#         for d in dict2:
#           if d not in dict1:
#             dict1[d] = []
#           dict1[d].append(dict2[d])
#         return dict1
#
#       if isinstance(noisy_in, list):
#         ret_list = [{}, {}]
#         for i in range(len(noisy_in)):
#           print i,
#           fdict = {noisy : noisy_in[i], dt : truth_in[i], sig_read : sig_in[i]}
#           output_out, filters_out = sess.run([output, filters], fdict)
#           ret_list[0] = dict_combine(ret_list[0], output2dict(output, output_out))
#           if filters is not []:
#             ret_list[1] = dict_combine(ret_list[1], output2dict(filters, filters_out))
#         print 'Done'
#
#       else:
#         fdict = {noisy : noisy_in, dt : truth_in, sig_read : sig_in}
#         output_out, filters_out = sess.run([output, filters], fdict)
#         ret_list = output2dict(output, output_out), output2dict(filters, filters_out)
#   return ret_list


def test_merge_simple_tt(log_dir, train_tensor, tt_mod=None, ret_filt=False, ret_grad=False):

  # First we split up the batch to make sure it's small enough to fit on a GTX 1080
  psize = 512
  bd = 64
  sh = train_tensor.shape

  if tt_mod is None:
    train_tensor = py_image.tensor2patches(train_tensor, psize, bd)
    print 'Traintensor resized from {} to {}'.format(sh, train_tensor.shape)
    pixlimit = (psize+2*bd)**2

    batchsize = (pixlimit-1) // np.prod(train_tensor.shape[1:3]) + 1
    numbatches = (train_tensor.shape[0]-1)//batchsize+1
    print 'With traintensor shape {}, using {} batches of length {} each'.format(
                        train_tensor.shape, numbatches, batchsize)
    tt = []
    for i in range(numbatches):
      tt.append(train_tensor[i*batchsize:(i+1)*batchsize,...])
    tt_mod = tt
  else:
    tt = tt_mod

  noisy_in = [t[...,:8] for t in tt]
  truth_in = [t[...,8] for t in tt]
  sig_in = [t[...,9:] for t in tt]

  g = tf.Graph()
  with g.as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

      ckpt_path = log_dir
      if 'model.ckpt' not in ckpt_path:
        ckpt_path = tf.train.latest_checkpoint(log_dir)

      if ckpt_path is not None:
        print 'Restoring from',ckpt_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        print 'Meta restored'
        saver.restore(sess, ckpt_path)
        print 'Weights restored'
      else:
        print 'No checkpoint found in {}'.format(ckpt_path)
        return None

      var_col = tf.get_collection('inputs')
      noisy = var_col[0]
      dt = var_col[1]
      sig_read = var_col[2]
      output_ = tf.get_collection('output')
      output = []
      for out in output_:
        if 'dnet' in out.name:
          output.append(out)
      filters_ = tf.get_collection('filters')
      filters = []
      for f in filters_:
        filters.append(f)


      if ret_grad:
        grad_stuff = []

        # vals = tf.get_collection(tf.GraphKeys.LOSSES)
        # for v in vals:
        #   print v
        # total_loss = tf.reduce_sum(vals)

        true_out = [out for out in output if 'noshow' not in out.name][0]
        print true_out.name
        total_loss = tf.reduce_mean(tf.square(true_out - dt))
        # total_loss = slim.losses.get_total_loss()
        loss_grad = tf.gradients(total_loss, noisy)[0]
#         lg_rel = tf.abs(loss_grad)
#         lg_rel = lg_rel / tf.reduce_mean(lg_rel, axis=-1, keep_dims=True)
#         lg_rel = tf.reduce_mean(lg_rel, axis=[1,2])
#         lg_mean = tf.abs(loss_grad)
#         lg_mean = tf.reduce_mean(lg_mean, axis=[1,2])
#
#         grad_stuff += [lg_rel, lg_mean]
#
#         lg_rel = tf.square(loss_grad)
#         lg_rel = lg_rel / tf.reduce_mean(lg_rel, axis=-1, keep_dims=True)
#         lg_rel = tf.reduce_mean(lg_rel, axis=[1,2])
#         lg_mean = tf.square(loss_grad)
#         lg_mean = tf.reduce_mean(lg_mean, axis=[1,2])
#
#         grad_stuff += [lg_rel, lg_mean]
#
#         grad_stuff = tf.stack(grad_stuff, axis=1)


      def output2dict(out_tf, out_np):
        ret = {}
        for i in range(len(out_tf)):
          ret[out_tf[i].name] = out_np[i]
        return ret

      def dict_combine(dict1, dict2):
        for d in dict2:
          if d not in dict1:
            dict1[d] = []
          dict1[d].append(dict2[d])
        return dict1

      ret_dict = {}
      filt_dict = {}
      grad_dict = {}
      to_run = {}
      to_run['output'] = output
      if ret_filt:
        to_run['filters'] = filters
      if ret_grad:
        to_run['grads'] = loss_grad

      for i in range(len(noisy_in)):
        print i,
        fdict = {noisy : noisy_in[i], dt : truth_in[i], sig_read : sig_in[i]}
        run_list = sess.run(to_run, fdict)
        output_out = run_list['output']
        ret_dict = dict_combine(ret_dict, output2dict(output, output_out))

        if ret_filt:
          filters_out = run_list['filters']
          filt_dict = dict_combine(filt_dict, output2dict(filters, filters_out))
        if ret_grad:
          grad_out = run_list['grads']
          grad_dict = dict_combine(grad_dict, {'grad' : grad_out})

      ret_dict = {k: np.concatenate(ret_dict[k], axis=0) for k in ret_dict}
      ret_dict = {k: py_image.patches2tensor(ret_dict[k], sh, psize, bd) for k in ret_dict}

      if ret_filt and filt_dict is not {}:
        filt_dict = {k: np.concatenate(filt_dict[k], axis=0) for k in filt_dict}
        filt_dict = {k: py_image.patches2tensor(filt_dict[k], sh, psize, bd) for k in filt_dict}

      if ret_grad and grad_dict is not {}:
        # gg = grad_dict['grad']
        # print 'grad stuff'
        # for g in gg:
        #   print g.shape
        grad_dict = {k: np.concatenate(grad_dict[k], axis=0) for k in grad_dict}
        grad_dict = {k: py_image.patches2tensor(grad_dict[k], sh, psize, bd) for k in grad_dict}


  return ret_dict, tt_mod, filt_dict, grad_dict



# Conv stuff

def make_conv2d_vars(in_tensor, W, K, name, stddev=.01):
    shape = [K, K, in_tensor.get_shape().as_list()[-1], W]
    initial = tf.truncated_normal(shape, stddev=stddev)
    filters = tf.Variable(initial, name=name + '_w')

    shape = [W]
    initial = tf.constant(0.0, shape=shape)
    bias = tf.Variable(initial, name=name+'_b')

    return filters, bias


# sres

def sres_upshape(x, n):
  ndims = len(x.get_shape().as_list())
  sh = tf.shape(x)
  if ndims==5:
    x = tf.reshape(x, [sh[0], sh[1], sh[2], n, n, sh[-1]])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [sh[0], sh[1]*n, sh[2]*n, sh[-1]])
  else:
    x = tf.reshape(x, [sh[0], sh[1], sh[2], n, n])
    x = tf.transpose(x, [0, 1, 3, 2, 4])
    x = tf.reshape(x, [sh[0], sh[1]*n, sh[2]*n])
  return x

def sres_downshape(x, n):
  ndims = len(x.get_shape().as_list())
  sh = tf.shape(x)
  if ndims==4:
    x = tf.reshape(x, [sh[0], sh[1]//n, n, sh[2]//n, n, sh[-1]])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [sh[0], sh[1]//n, sh[2]//n, n*n, sh[-1]])
  else:
    x = tf.reshape(x, [sh[0], sh[1]//n, n, sh[2]//n, n])
    x = tf.transpose(x, [0, 1, 3, 2, 4])
    x = tf.reshape(x, [sh[0], sh[1]//n, sh[2]//n, n*n])
  return x


# optimal convolve

def solve_convolve(noisy, truth, final_K, excl_edges=False):
  kpad = final_K//2
  ch = noisy.get_shape().as_list()[-1]
  ch1 = truth.get_shape().as_list()[-1]
  sh = tf.shape(noisy)
  h, w = sh[1], sh[2]
  img_stack = []
  noisy = tf.pad(noisy, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]])
  for i in range(final_K):
    for j in range(final_K):
      img_stack.append(noisy[:, i:h+i, j:w+j, :])
  img_stack = tf.stack(img_stack, axis=-2)
  is0 = img_stack
  if excl_edges:
    img_stack = img_stack[:, kpad:-kpad, kpad:-kpad, :]
    truth = truth[:, kpad:-kpad, kpad:-kpad]
    h = h - 2*kpad
    w = w - 2*kpad
  A = tf.reshape(img_stack, [tf.shape(img_stack)[0], h*w, final_K**2 * ch])
  b = tf.reshape(truth, [tf.shape(truth)[0], h*w, ch1])
  x_ = tf.matrix_solve_ls(A, b, fast=False)
  x = tf.reshape(x_, [tf.shape(truth)[0], final_K, final_K, ch, ch1])
  return x


def convolve(img_stack, filts, final_K, final_W):
  initial_W = img_stack.get_shape().as_list()[-1]

  fsh = tf.shape(filts)
  filts = tf.reshape(filts, [fsh[0], fsh[1], fsh[2], final_K ** 2 * initial_W, final_W])

  kpad = final_K//2
  imgs = tf.pad(img_stack, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]])
  ish = tf.shape(img_stack)
  img_stack = []
  for i in range(final_K):
    for j in range(final_K):
      img_stack.append(imgs[:, i:tf.shape(imgs)[1]-2*kpad+i, j:tf.shape(imgs)[2]-2*kpad+j, :])
  img_stack = tf.stack(img_stack, axis=-2)
  img_stack = tf.reshape(img_stack, [ish[0], ish[1], ish[2], final_K**2 * initial_W, 1])
  img_net = tf.reduce_sum(img_stack * filts, axis=-2) # removes the final_K**2*initial_W dimension but keeps final_W
  return img_net

def optimal_convolve(noisy, truth, final_K, conv_stack=None):
  filts = solve_convolve(noisy, truth, final_K, True)

  fsh = tf.shape(filts)
  filts_ = tf.expand_dims(tf.expand_dims(filts, axis=1), axis=1)
  final_W = truth.get_shape().as_list()[-1]

  if conv_stack is None:
    conv_stack = noisy
  shift1 = convolve(conv_stack, filts_, final_K, final_W)
  return shift1, filts


# For separable stuff
def convolve_aniso(img_stack, filts, final_Kh, final_Kw, final_W, layerwise=False):
  initial_W = img_stack.get_shape().as_list()[-1]

  fsh = tf.shape(filts)
  if layerwise:
    filts = tf.reshape(filts, [fsh[0], fsh[1], fsh[2], final_Kh * final_Kw,           initial_W])
  else:
    filts = tf.reshape(filts, [fsh[0], fsh[1], fsh[2], final_Kh * final_Kw * initial_W, final_W])

  kpadh = final_Kh//2
  kpadw = final_Kw//2
  imgs = tf.pad(img_stack, [[0,0],[kpadh,kpadh],[kpadw,kpadw],[0,0]])
  ish = tf.shape(img_stack)
  img_stack = []
  for i in range(final_Kh):
    for j in range(final_Kw):
      img_stack.append(imgs[:, i:tf.shape(imgs)[1]-2*kpadh+i, j:tf.shape(imgs)[2]-2*kpadw+j, :])
  img_stack = tf.stack(img_stack, axis=-2)
  if layerwise:
    img_stack = tf.reshape(img_stack, [ish[0], ish[1], ish[2], final_Kh * final_Kw, initial_W])
  else:
    img_stack = tf.reshape(img_stack, [ish[0], ish[1], ish[2], final_Kh * final_Kw * initial_W, 1])
  img_net = tf.reduce_sum(img_stack * filts, axis=-2) # removes the final_K**2*initial_W dimension but keeps final_W
  return img_net

# Helper

def tf_fn_test(tf_fn):
  def ret_fn(*args):
    g = tf.Graph()
    with g.as_default():
      tf_args = []
      for arg in args:
        tf_args.append(tf.placeholder(tf.float32, shape=arg.shape))
      output = tf_fn(*tf_args)
      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:
        sess.run(init_op)
        fdict = {tf_arg : np_arg for tf_arg, np_arg in zip(args, tf_args)}
        output = sess.run(output, feed_dict=fdict)
        return output
  return ret_fn

# HDR Plus

def rcwindow(N):
    x = tf.linspace(0., N, N+1)[:-1]
    rcw = .5 - .5 * tf.cos(2.*np.pi * (x + .5) / N)
    rcw = tf.reshape(rcw,(N,1)) * tf.reshape(rcw,(1,N))
    return rcw


def roll_tf(x, shift, axis=0):
    sh = tf.shape(x)
    n = sh[axis]
    shift = shift % n
    bl0 = tf.concat([sh[:axis], [n-shift], sh[axis+1:]], axis=0)
    bl1 = tf.concat([sh[:axis], [shift],   sh[axis+1:]], axis=0)
    or0 = tf.concat([tf.zeros_like(sh[:axis]), [shift], tf.zeros_like(sh[axis+1:])], axis=0)
    or1 = tf.zeros_like(bl0)
    x0 = tf.slice(x, or0, bl0)
    x1 = tf.slice(x, or1, bl1)
    return tf.concat([x0, x1], axis=axis)


def hdrplus_merge(imgs, N, c, sig):
    ccast_tf = lambda x : tf.complex(x, tf.zeros_like(x))

    # imgs is [batch, h, w, ch]
    rcw = tf.expand_dims(rcwindow(N), axis=-1)
    imgs = imgs * rcw
    imgs = tf.transpose(imgs, [0, 3, 1, 2])
    imgs_f = tf.fft2d(ccast_tf(imgs))
    imgs_f = tf.transpose(imgs_f, [0, 2, 3, 1])
    Dz2 = tf.square(tf.abs(imgs_f[...,0:1] - imgs_f))
    Az = Dz2 / (Dz2 + c*sig**2)
    filt0 = 1 + tf.expand_dims(tf.reduce_sum(Az[...,1:], axis=-1), axis=-1)
    filts = tf.concat([filt0, 1 - Az[...,1:]], axis=-1)
    output_f = tf.reduce_mean(imgs_f * ccast_tf(filts), axis=-1)
    output_f = tf.real(tf.ifft2d(output_f))

    return output_f


def hdrplus_tiled(noisy, N, sig, c=10**2.25):
    sh = tf.shape(noisy)[0:3]
    buffer = tf.zeros_like(noisy[...,0])
    allpics = []
    for i in range(2):
        for j in range(2):
            nrolled = roll_tf(roll_tf(noisy, shift=-N//2*i, axis=1), shift=-N//2*j, axis=2)
            hpatches = (tf.transpose(tf.reshape(nrolled, [sh[0], sh[1]//N, N, sh[2]//N, N, -1]), [0,1,3,2,4,5]))
            hpatches = tf.reshape(hpatches, [sh[0]*sh[1]*sh[2]//N**2, N, N, -1])
            merged = hdrplus_merge(hpatches, N, c, sig)
            merged = tf.reshape(merged, [sh[0], sh[1]//N, sh[2]//N, N, N])
            merged = (tf.reshape(tf.transpose(merged, [0,1,3,2,4]), sh))
            merged = roll_tf(roll_tf(merged, axis=1, shift=N//2*i), axis=2, shift=N//2*j)
            buffer += merged
            allpics.append(merged)
    return buffer
