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
"""Trains a KPN model.

See the README.md file for compilation and running instructions.
"""

import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile

import kpn_arch
import kpn_data_provider
from demosaic_utils import *
from tf_image import *

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 4, 'The number of images in each batch.')

flags.DEFINE_integer('patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_integer('burst_size', 8, 'The number of images in each burst.')

flags.DEFINE_string('train_log_dir', '/tmp/kpn_logs/',
                    'Directory where to write training.')

flags.DEFINE_string('data_dir', 'data/train', '')

flags.DEFINE_string('dataset_dir', 'open-images-dataset', '')

flags.DEFINE_float('learning_rate', .0001, 'The learning rate')

flags.DEFINE_float('anneal', .9998, 'Anneal rate')


flags.DEFINE_integer('max_number_of_steps', 500000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('use_noise', 1,
                     '1/0 use noise.')

flags.DEFINE_integer('real_data', 0,
                     'synthetic or real.')

flags.DEFINE_integer('pre_rnn', 1,
                     'pre rnn .')

flags.DEFINE_float('filt_sup', -3.,
                     'filter bank supervision, -3 for none')

flags.DEFINE_string('layer_type', 'singlestd',
                    'Layers in singlestd.')

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = flags.FLAGS



# Makes test data that looks like training data, can use for validation set
def train_merge_simple_get_synth_test(log_dir):
  g = tf.Graph()
  with g.as_default():

    batch = FLAGS.batch_size
    BURST_LENGTH = FLAGS.burst_size
    height = width = FLAGS.patch_size

    dataset_dir = os.path.join(FLAGS.dataset_dir, 'validation')
    truth = kpn_data_provider.load_batch_hqjitter(dataset_dir, patches_per_img=4, min_queue=4,
                        BURST_LENGTH=BURST_LENGTH, batch_size=batch, repeats=2, height=height, width=width, degamma=2.2, to_shift=1.,
                        upscale=4, jitter=16, smalljitter=2)
    truth = tf.reduce_mean(truth, axis=-2)

    degamma = 1.
    white_level = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], np.log10(.1), np.log10(1.)))
    truth = (white_level * truth ** degamma)
    sig_read = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], -3., -1.5))
    sig_shot = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], -2., -1.))
    noisy, _ = add_read_shot_tf(truth, sig_read, sig_shot)
    sig_read_single_std = tf.sqrt(sig_read**2 + tf.maximum(0., noisy[...,0:1]) * sig_shot**2)

    full_batch = {
        'truth' : truth,
        'noisy' : noisy,
        'sig_read' : tf.squeeze(sig_read),
        'sig_shot' : tf.squeeze(sig_shot),
        'white_level' : tf.squeeze(white_level)
    }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      print 'Initializers'
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      print 'Thread coordinator'
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      max_steps = FLAGS.max_number_of_steps
      for i_step in range(max_steps):
        print i_step
        batch_np = sess.run(full_batch)
        filename = 'synthetic_{}x{}x{}x{}_num{:03d}.npz'.format(batch, height, width, BURST_LENGTH, i_step)
        full_path = os.path.join(log_dir, filename)
        np.savez(full_path, **batch_np)


      coord.request_stop()
      coord.join(threads)


# Actual training function
def train_merge_simple(filenames):
  g = tf.Graph()
  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks)):

      batch = FLAGS.batch_size
      BURST_LENGTH = FLAGS.burst_size
      height = width = FLAGS.patch_size
      # gs = tf.placeholder(tf.int32, shape=(), name="gs")
      gs = tf.Variable(0,name='global_step',trainable=False)

      synthetic = True
      color = False

      # Data preprocessing. Tons of junk here
      if synthetic:

        dataset_dir = os.path.join(FLAGS.dataset_dir, 'train')
        demosaic_truth = kpn_data_provider.load_batch_hqjitter(dataset_dir, patches_per_img=4, min_queue=16,
                            BURST_LENGTH=BURST_LENGTH, batch_size=batch, repeats=2, height=height, width=width,
                            degamma=2.2, to_shift=1., upscale=4, jitter=16, smalljitter=2)
        shift = tf.zeros([1])
        if not color:
          demosaic_truth = tf.reduce_mean(demosaic_truth, axis=-2)

        truth_all = demosaic_truth

#         sig_read = tf.random_uniform([batch, 1, 1, 1], 0.005, 0.1)
#         noisy = tf.clip_by_value(demosaic_truth + tf.random_normal([batch, height, width, BURST_LENGTH]) * sig_read, 0., 1.)
#         demosaic_truth = demosaic_truth[...,0]

        # demosaic_truth = tf.tile(demosaic_truth[...,0:1], [1,1,1,BURST_LENGTH])

        degamma = 1.
        white_level = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], np.log10(.1), np.log10(1.)))
        if color:
          white_level = tf.expand_dims(white_level, axis=-1)
        demosaic_truth = (white_level * demosaic_truth ** degamma)

        sig_read = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], -3., -1.5))
        sig_shot = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], -2., -1.))
        # sig_read = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], -3., -2))
        # sig_shot = tf.pow(10., tf.random_uniform([batch, 1, 1, 1], -2., -1.5))
        dec = decimatergb(demosaic_truth, BURST_LENGTH) if color else demosaic_truth
        print 'DEC',dec.get_shape().as_list()
        noisy_, _ = add_read_shot_tf(dec, sig_read, sig_shot)
        # noisy_ = dec # TAKE OUT TO RETURN THE NOISEEE
        print 'NOISY_',noisy_.get_shape().as_list()
        print 'DT2',demosaic_truth.get_shape().as_list()
        # noisy = tf.clip_by_value(noisy_, 0., 1.)
        noisy = noisy_

        # noisy = tf.image.resize_images(demosaic_truth, [height//2, width//2], method=tf.image.ResizeMethod.AREA)
        # noisy = demosaic_truth
        # noisy = .25 * (noisy[:,::2,::2,:]+noisy[:,1::2,::2,:]+noisy[:,::2,1::2,:]+noisy[:,1::2,1::2,:])
        # noisy = .25 * (noisy[:,::2,::2,:]+noisy[:,1::2,::2,:]+noisy[:,::2,1::2,:]+noisy[:,1::2,1::2,:])

        first2 = demosaic_truth[...,:2]
        demosaic_truth = demosaic_truth[...,0,:] if color else demosaic_truth[...,0]

        # demosaic_truth = sres_upshape(sres_out, 2)

        print 'DT3',demosaic_truth.get_shape().as_list()

        if color:
          print 'NOISY pre',noisy.get_shape().as_list()
          noisy = small_bayer_stack(noisy)
          white_level = white_level[...,0]
          dumb0 = dumb_tf_demosaic(tf22reshape(noisy[...,::BURST_LENGTH]))
          dumb_avg = dumb_tf_demosaic(tf22reshape(tf.reduce_mean(tf.reshape(noisy, [tf.shape(noisy)[0],tf.shape(noisy)[1],tf.shape(noisy)[2],4,BURST_LENGTH]), axis=-1)))


      else:
        # burst, merged, demosaic, readvar, shotfactor = kpn_data_provider.inputs_patches(filenames, FLAGS.batch_size, FLAGS.burst_size)

        print('Have {} filenames, first is {}'.format(len(filenames), filenames[0]))
        burst, demosaic, readvar, shotfactor, _ = \
          kpn_data_provider.inputs(filenames, batch_size=batch, height=height*2, width=width*2, depth=64, burst_length=BURST_LENGTH)

        noisy = batch_down2(burst)
        demosaic_truth = batch_down2(tf.reduce_mean(demosaic, axis=-1))
        truth_all = tf.expand_dims(demosaic_truth, axis=-1)
        shift = tf.zeros([batch, BURST_LENGTH-1])
        noisiness = tf.reshape(tf.reduce_mean(readvar,axis=1),[-1,1,1,1]) + tf.maximum(0.,noisy[...,0:1]) * tf.reshape(tf.reduce_mean(shotfactor,axis=1),[-1,1,1,1])
        sig_read = tf.reshape(tf.sqrt(tf.reduce_mean(noisiness, axis=[1,2,3])), [batch,1,1,1])
        sig_shot = sig_read
        sig_read = tf.sqrt(noisiness)
        white_level = tf.reduce_max(tf.reshape(demosaic_truth, [batch, -1]), axis=-1)
        white_level = tf.reshape(white_level, [batch,1,1])
        # white_level = tf.ones([batch, 1, 1])
        if color:
          noisy = small_bayer_stack(burst)
          demosaic_truth = demosaic
          # dumb0 = dumb_tf_demosaic(burst[...,0])
          # dumb_avg = dumb_tf_demosaic(tf.reduce_mean(burst, axis=-1))
          dumb0 = dumb_tf_demosaic(tf22reshape(noisy[...,::BURST_LENGTH]))
          dumb_avg = dumb_tf_demosaic(tf22reshape(tf.reduce_mean(tf.reshape(noisy, [tf.shape(noisy)[0],tf.shape(noisy)[1],tf.shape(noisy)[2],4,BURST_LENGTH]), axis=-2)))
          white_level = tf.ones([batch, 1, 1, 1])

          
      dt = demosaic_truth
      print 'DT4',demosaic_truth.get_shape().as_list()
      print 'DT5',dt.get_shape().as_list()
      nt = noisy

      sig_read = tf.tile(sig_read, [1, tf.shape(noisy)[1], tf.shape(noisy)[2], 1])
      sig_shot = tf.tile(sig_shot, [1, tf.shape(noisy)[1], tf.shape(noisy)[2], 1])
      sig_tower = tf.concat([sig_shot, sig_read], axis=-1)

      noisy = tf.placeholder_with_default(noisy, [None, None, None, BURST_LENGTH], name='noisy')
      dt = tf.placeholder_with_default(dt, [None, None, None], name='dt')
      sig_tower = tf.placeholder_with_default((sig_tower), [None, None, None, 2], name='sig_tower')

      tf.add_to_collection('inputs', noisy)
      tf.add_to_collection('inputs', dt)
      tf.add_to_collection('inputs', sig_tower)
      print 'Added to collection'

      sig_shot = sig_tower[...,0:1]
      sig_read = sig_tower[...,1:2]
      sig_read_single_std = tf.sqrt(sig_read**2 + tf.maximum(0., noisy[...,0:1]) * sig_shot**2)
      sig_read_dual_params = tf.concat([sig_read, sig_shot], axis=-1)
      sig_read_empty = tf.zeros_like(noisy[...,0:0])

      sig_reads = {
          'singlestd' : sig_read_single_std,
          'dualparams' : sig_read_dual_params,
          'empty' : sig_read_empty
      }

      sig_read = sig_reads[FLAGS.layer_type]

      plots = {}
      max_out = 4
      image_summaries = []

      new_stack = None

      # Silly baselines for numerical and visual comparison

      dumb = {}
      if color:
        dumb['dumb0'] = dumb0
        dumb['dumb-avg'] = dumb_avg
        dumb['simple'] = tf.image.resize_images(tf.stack([noisy[...,0],.5*(noisy[...,BURST_LENGTH]+noisy[...,2*BURST_LENGTH]),
                                                          noisy[...,3*BURST_LENGTH]],axis=-1),tf.shape(noisy)[1:3]*2)
      else:
        dumb['dumb0'] = noisy[...,0] # noisy[...,0]
        dumb['dumb-avg'] = tf.reduce_mean(noisy, axis=-1)
        dhdr = []
        # Below is dangerous since it hard codes a loop size w.r.t. batch
        for i in range(batch):
          dhdr.append(hdrplus_tiled(noisy[i:i+1,...], N=16, sig=tf.reduce_mean(sig_read_single_std[i,...]), c=10**2.25))
        dumb['dumbhdr'] = tf.concat(dhdr, axis=0)

        ### dumb methods for superres
        # dumb['dumb0'] = tf.tile(sres_in[...,0:1], [1, 1, 1, 4])
        # dumb['dumb0bilrp'] = tf.image.resize_images(sres_in[...,0:1], [height, width])[...,0]

        # dumb['dumb0'] = tf.image.resize_images(noisy[...,0:1], [height, width], method=tf.image.ResizeMethod.BICUBIC)[...,0]
        # dumb['dumb-avg'] = tf.image.resize_images(tf.expand_dims(tf.reduce_mean(noisy, axis=-1), axis=-1),
        #                                           [height, width], method=tf.image.ResizeMethod.BICUBIC)[...,0]


      # if synthetic and not color:
      #   shift_mask = tf.tile(tf.concat([tf.ones([batch, 1, 1, 1]), tf.reshape(1. - shift, [batch, 1, 1, BURST_LENGTH-1])], axis=-1), [1, height, width, 1])
      #   m_mask = noisy * shift_mask
      #   # dumb['dumb-M-avg'] = tf.reduce_sum(noisy * shift_mask, axis=-1) / tf.reduce_sum(shift_mask, axis=-1)
      m_mask = noisy

      demosaic = {}
      filters = {}
      anneals = {}
      # if not color:
      #   demosaic['dhdr'] = merge.hdrplus_merge(noisy, sig_read, c=4e4)
      dnet = 'dnet-'
      with tf.variable_scope('generator'):

        noisy_sig = tf.concat([noisy, sig_read], axis=-1)
        final_K = 5 # kpn filter output size
        final_W = 1 # number of dims in output
        fh, fw = 2, 4 # how to show the filters in tensorboard
        invert_preprocessing = True
        filt_sup = False
        lbuff = 8 # crop this out when reporting psnr
        wlb = tf.reshape(white_level, [batch, 1,1,1])

        # Use this to gamma correct, etc. for taking loss
        def invert_preproc(imgs):
          wl = tf.reshape(white_level, [-1])
          return sRGBforward(tf.transpose(tf.transpose(imgs) / wl))[:, lbuff:-lbuff, lbuff:-lbuff, ...]

        optfilt_on = False
        if optfilt_on:
          # Least squares optimal filter
          dumb['optimalA'], optfilt = optimal_convolve(invert_preproc(noisy),
                                                     invert_preproc(tf.expand_dims(dt, axis=-1)), final_K=final_K, conv_stack=noisy)


        use_S1 = True
        if (use_S1):
          key = dnet + 's1'
          # KPN created here
          demosaic[key], filters[key] = kpn_arch.convolve_net2(noisy_sig, noisy, final_K, final_W,
                                                            ch0=64, N=2, D=3,
                                                            scope='cnet2asep', separable=False, bonus=False, avg_spatial=False)


          meanfilt1 = tf.reduce_mean(filters[key], axis=[1,2])
          meanfilt1_ = tf.concat([tf.zeros_like(meanfilt1[...,0:1,:]), meanfilt1[...,1:,:]], axis=-2)

          demosaic[key] = demosaic[key][...,0]

          # Annealed loss term
          anneal = FLAGS.anneal
          if anneal > 0:
            per_layer = kpn_arch.convolve_per_layer(noisy, filters[key], final_K, final_W)
            for ii in range(BURST_LENGTH):
              itmd = per_layer[...,ii] * BURST_LENGTH
              # If we include image in demosaic dictionary the loss will be applied
              demosaic[dnet + 'da{}_noshow'.format(ii)] = itmd
              # alpha hardcoded as 10^2, beta set with FLAGS.anneal
              anneal_coeff = tf.pow(anneal, tf.cast(gs, tf.float32)) * (10. ** (2))
              anneals[dnet + 'da{}_noshow'.format(ii)] = anneal_coeff
              # Tensorboard junk
              if ii==0:
                astr = str(anneal)
                astr = astr[astr.find('.')+1:]
                plots = store_plot(plots, 'anneal/anneal', tf.log(anneal_coeff)/tf.log(10.),'a{}'.format(astr))
              if ii < 2:
                itmd_loss = tf.reduce_mean(tf.square(invert_preproc(itmd) - invert_preproc(dt)))
                plots = store_plot(plots, 'itmds/psnr', -10.*tf.log(itmd_loss)/tf.log(10.), 'da{}'.format(ii))

        # Necessary hooks for evaluating without reconstructing entire graph
        d_all_unproc = dict(dumb.items() + demosaic.items())
        for k in d_all_unproc:
          temp_tensor = tf.identity(d_all_unproc[k], name=k)
          tf.add_to_collection('output', temp_tensor)
        for k in filters:
          temp_tensor = tf.identity(filters[k], name=k)
          tf.add_to_collection('filters', temp_tensor)


        # Run gamma correction
        for d in dumb:
          dumb[d] = invert_preproc(dumb[d])
        for d in demosaic:
          demosaic[d] = invert_preproc(demosaic[d])
        dt = invert_preproc(dt)

        # Tensorboard stuff for filters
        fh, fw = 2, 4
        if optfilt_on:
          image_summaries.append(tf.summary.image('optfiltA', filts2imgs(optfilt,fh,fw), max_outputs=max_out))

        if use_S1:
          image_summaries.append(tf.summary.image('meanfilt1', filts2imgs(meanfilt1,fh,fw), max_outputs=max_out))
          image_summaries.append(tf.summary.image('meanfilt1_', filts2imgs(meanfilt1_,fh,fw), max_outputs=max_out))
          if optfilt_on:
            floss1 = tf.reduce_mean(tf.abs(meanfilt1 - optfilt))
            fs = FLAGS.filt_sup
            if fs < -2.:
              fs = 0
            else:
              fs = 10.**fs
            if fs > 0:
              # No longer used anywhere - supervision against the optimal filter
              slim.losses.add_loss(floss1 * fs)
            plots = store_plot(plots, 'floss/floss', tf.log(floss1)/tf.log(10.), 'f1e{}'.format(FLAGS.filt_sup))


      # Actually calculate image losses
      d_all = dict(dumb.items() + demosaic.items())

      losses = []
      for d in demosaic:
        if d.startswith(dnet):
          print 'LOSSES for',d
          a = 1.
          if anneals is not None and d in anneals:
            a = anneals[d]
            print 'includes anneal'
          losses.append(basic_img_loss(demosaic[d], dt) * a)
      slim.losses.add_loss(tf.reduce_sum(tf.stack(losses)))


      total_loss = slim.losses.get_total_loss()

      # Lots of tensorboard stuff

      plots = store_plot(plots, 'loss/log10total', tf.log(total_loss)/tf.log(10.))


      # PSNR comparisons
      psnrs_g = {}
      for d in demosaic:
        psnrs_g[d] = psnr_tf_batch((demosaic[d]), (dt))
      psnrs = {}
      for d in dumb:
        psnrs[d] = psnr_tf_batch((dumb[d]), (dt))



      # Create some summaries to visualize the training process:
      gamma = 1./1
      disp_wl = 1
#       white_level = tf.ones_like(white_level)

      image_summaries.append(tf.summary.image('diffs/base', process_for_tboard(.5 + (dumb['dumb0']-dt)/disp_wl, gamma=gamma), max_outputs=max_out))

      for d in psnrs_g:
        if 'noshow' not in d:
          image_summaries.append(tf.summary.image('demosaic/'+d, process_for_tboard(demosaic[d]/disp_wl, gamma=gamma), max_outputs=max_out))
          image_summaries.append(tf.summary.image('diffs/'+d, process_for_tboard(.5 + (demosaic[d]-dt)/disp_wl, gamma=gamma), max_outputs=max_out))

      pref = psnr_tf_batch(dumb['dumb0'], dt)
      if not color:
        sref = tf_ssim(tf.expand_dims(dumb['dumb0'],axis=-1), tf.expand_dims(dt,axis=-1))
      for d in sorted(d_all):
        if 'noshow' not in d:
          plots = store_plot(plots, 'plot/psnrs', psnr_tf_batch(d_all[d], dt), d)
          plots = store_plot(plots, 'dplot/psnrs', psnr_tf_batch(d_all[d], dt)-pref, d)
          if not color:
            plots = store_plot(plots, 'plot/ssim', tf_ssim(tf.expand_dims(d_all[d],axis=-1), tf.expand_dims(dt,axis=-1)), d)
            plots = store_plot(plots, 'dplot/ssim', tf_ssim(tf.expand_dims(d_all[d],axis=-1), tf.expand_dims(dt,axis=-1))-sref, d)

      image_summaries.append(tf.summary.image('demosaic_truth_0', process_for_tboard(dt/disp_wl, gamma=gamma), max_outputs=max_out))

      for d in dumb:
#         tf.summary.scalar('psnrs/psnr_' + d, psnrs[d])
        image_summaries.append(tf.summary.image('dumb/' + d, process_for_tboard(dumb[d]/disp_wl, gamma=gamma), max_outputs=max_out))

      if synthetic:
        if color:
          image_summaries.append(tf.summary.image('truths/avg', process_for_tboard(tf.reduce_mean(truth_all, axis=-2), gamma=1.), max_outputs=max_out))
          for i in range(BURST_LENGTH):
            image_summaries.append(tf.summary.image('truths/m' + str(i), process_for_tboard(truth_all[...,i,:], gamma=1.), max_outputs=max_out))
        else:
          image_summaries.append(tf.summary.image('truths/avg', process_for_tboard(tf.expand_dims(tf.reduce_mean(truth_all, axis=-1), axis=-1), gamma=1.), max_outputs=max_out))
          for i in range(BURST_LENGTH):
            image_summaries.append(tf.summary.image('truths/m' + str(i), process_for_tboard(truth_all[...,i:i+1], gamma=1.), max_outputs=max_out))


      g_index = tf.placeholder(tf.int32, shape=(), name="g_index")
      summaries = gen_plots(plots, g_index)
      image_summaries = tf.summary.merge(image_summaries)


      # Actual optimizer
      g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_step_g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(total_loss, global_step=gs, var_list=g_vars)

      # Just including this for fun, could use in future
      do_vgg = False
      if (do_vgg):
        vgg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')
        vgg_saver = tf.train.Saver(var_list=vgg_vars)

      # Real computation below
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:
        num_writers = max([len(plots[n]) for n in plots])
        print 'Starting', num_writers, 'writers'
        writers = [tf.summary.FileWriter(FLAGS.train_log_dir + '/writer' + str(i), sess.graph) for i in range(num_writers)]
        saver = tf.train.Saver(max_to_keep=None)

        print 'Initializers'
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # This loads from last checkpoint, necessary for borg training to
        # not get restarted at every preemption event
        ckpt_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
        if ckpt_path is not None:
          print 'Restoring from',ckpt_path
          saver.restore(sess, ckpt_path)


        print 'Thread coordinator'
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        max_steps = FLAGS.max_number_of_steps
        for i_step in range(max_steps):
          #run optimization step5
          _, loss, i = sess.run([train_step_g, total_loss, gs])

          print 'Step',i,'loss =',loss
          #training set summaries for tensorboard
          if True and (((i+1)%10 == 0 and i < 200) or ((i+1)%100 == 0)):
            print 'Writing summary at step',i
            tf_vars = [sig_read, demosaic_truth, shift, noisy, dt, white_level, sig_read, sig_shot, truth_all]
            np_vals = sess.run(tf_vars, feed_dict={g_index : 0})
            fdict = {tf_var : np_val for tf_var, np_val in zip(tf_vars, np_vals)}
            run_summaries(sess, fdict, writers, summaries, g_index, step=i)
            if ((i+1)%10 == 0 and i < 200) or ((i+1)%200 == 0):
              fdict[g_index] = 0
              imgs, = sess.run([image_summaries], feed_dict=fdict)
              writers[0].add_summary(imgs, i)
          if (i+1)%2000 == 0:
            print 'Saving ckpt at step',i
            saver.save(sess, FLAGS.train_log_dir + 'model.ckpt', global_step=i)
        for w in writers:
          w.close()

        coord.request_stop()
        coord.join(threads)





def main(_):
  if not gfile.Exists(FLAGS.train_log_dir):
    gfile.MakeDirs(FLAGS.train_log_dir)

  data_dir = FLAGS.data_dir
  filenames = [os.path.join(data_dir, f) for f in gfile.ListDirectory(data_dir) if f.startswith('tfdata')]
  print(len(filenames),'real data files found (one whole burst each)')

  train_merge_simple(filenames)




if __name__ == '__main__':
  app.run()
