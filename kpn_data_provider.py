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
"""Contains code for loading and preprocessing the MNIST data."""

import os
import tensorflow as tf
slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider
dataset = slim.dataset
queues = slim.queues
gfile = tf.gfile
import demosaic_utils


def make_demosaic(image, height, width, BURST_LENGTH, to_shift, upscale, jitter):
  y = tf.random_uniform([1], jitter * upscale, tf.shape(image)[0]-height - jitter*upscale, tf.int32)
  x = tf.random_uniform([1], jitter * upscale, tf.shape(image)[1]-width - jitter*upscale, tf.int32)
  y, x = y[0], x[0]
  demosaic = tf.reshape(image[y:y+height, x:x+width, :], (1, height, width, 1, 3))
  delta = tf.random_uniform([BURST_LENGTH-1,2], -jitter*upscale, jitter*upscale+1, tf.int32)
  # delta_big = tf.random_uniform([BURST_LENGTH-1,2], -20, 20, tf.int32)
  shift_mask = tf.random_uniform([BURST_LENGTH-1, 1], 0., BURST_LENGTH-1., tf.float32) * to_shift
  shift_mask = tf.where(shift_mask > BURST_LENGTH-2., tf.ones([BURST_LENGTH-1, 1]), tf.zeros([BURST_LENGTH-1, 1]))
  delta = delta # + tf.cast(tf.tile(shift_mask, [1, 2]), tf.int32) * delta_big
  shift_mask = tf.reshape(shift_mask, [1, BURST_LENGTH-1])
  for d in range(BURST_LENGTH-1):
    y_ = (y + delta[d,0]) # % (tf.shape(image)[0]-height)
    x_ = (x + delta[d,1]) # % (tf.shape(image)[1]-width)
    demosaic2 = tf.reshape(image[y_:y_+height, x_:x_+width, :], (1, height, width, 1, 3))
    demosaic = tf.concat([demosaic, demosaic2], axis=3)
  return demosaic, shift_mask


def make_stack_demosaic(image, height, width, depth, BURST_LENGTH, to_shift, upscale, jitter):
  height = height * upscale
  width = width * upscale
  v_error = tf.maximum(height + 2 * jitter * upscale - tf.shape(image)[0] + 1, 0)
  h_error = tf.maximum(width + 2 * jitter * upscale - tf.shape(image)[1] + 1, 0)
  image = tf.pad(image, [[0,v_error],[0,h_error],[0,0]])

  demosaic_stack, shift_stack = make_demosaic(image, height, width, BURST_LENGTH, to_shift, upscale, jitter)
  for i in range(depth-1):
    demosaic, shift_mask = make_demosaic(image, height, width, BURST_LENGTH, to_shift, upscale, jitter)
    demosaic_stack = tf.concat((demosaic_stack, demosaic), axis=0)
    shift_stack = tf.concat((shift_stack, shift_mask) , axis=0)

  dt = tf.reshape(tf.transpose(demosaic_stack, [0, 3, 1, 2, 4]), [-1, height, width, 3])
  height = height // upscale
  width = width // upscale
  dt = tf.image.resize_images(dt, [height, width], method=tf.image.ResizeMethod.AREA)
  demosaic_stack = tf.transpose(tf.reshape(dt, [depth, BURST_LENGTH, height, width, 3]), [0, 2, 3, 1, 4])

  mosaic = tf.stack((demosaic_stack[:,::2,::2,:,0],demosaic_stack[:,::2,1::2,:,1],demosaic_stack[:,1::2,::2,:,1],demosaic_stack[:,1::2,1::2,:,2]), axis=-1)
  mosaic = demosaic_utils.tf22reshape2(mosaic, BURST_LENGTH)
  mosaic = tf.reshape(mosaic, (depth, height, width, BURST_LENGTH))
  return mosaic, demosaic_stack, shift_stack


def load_batch_demosaic(BURST_LENGTH, dataset_dir, batch_size=32, height=64, width=64, degamma=1., to_shift=1., upscale=1, jitter=1):

  filenames = [os.path.join(dataset_dir, f) for f in gfile.ListDirectory(dataset_dir)]
  filename_queue = tf.train.string_input_producer(filenames)

  mosaic = None
  while mosaic == None:
    _, image_file = tf.WholeFileReader().read(filename_queue)
    image = tf.image.decode_image(image_file)
    mosaic, demosaic, shift = make_stack_demosaic((tf.cast(image[0], tf.float32) / 255.)**degamma,
                                                  height, width, 128, BURST_LENGTH, to_shift, upscale, jitter)

  # Batch it up.
  mosaic, demosaic, shift = tf.train.shuffle_batch(
        [mosaic, demosaic, shift],
        batch_size=batch_size,
        num_threads=2,
        capacity=500 + 3 * batch_size,
        enqueue_many=True,
        min_after_dequeue=100)

  return mosaic, demosaic, shift


def make_batch_hqjitter(patches, BURST_LENGTH, batch_size, repeats, height, width,
                        to_shift, upscale, jitter, smalljitter):
  # patches is [BURST_LENGTH, h_up, w_up, 3]
  j_up = jitter * upscale
  h_up = height * upscale # + 2 * j_up
  w_up = width * upscale # + 2 * j_up

  bigj_patches = patches
  delta_up = (jitter - smalljitter) * upscale
  smallj_patches = patches[:, delta_up:-delta_up, delta_up:-delta_up, ...]

  unique = batch_size//repeats
  batch = []
  for i in range(unique):
    for j in range(repeats):
      curr = [patches[i, j_up:-j_up, j_up:-j_up, :]]
      prob = tf.minimum(tf.cast(tf.random_poisson(1.5, []), tf.float32)/BURST_LENGTH, 1.)
      for k in range(BURST_LENGTH - 1):
        flip = tf.random_uniform([])
        p2use = tf.cond(flip < prob, lambda : bigj_patches, lambda : smallj_patches)
        curr.append(tf.random_crop(p2use[i, ...], [h_up, w_up, 3]))
      curr = tf.stack(curr, axis=0)
      curr = tf.image.resize_images(curr, [height, width], method=tf.image.ResizeMethod.AREA)
      curr = tf.transpose(curr, [1,2,3,0])
      batch.append(curr)
  batch = tf.stack(batch, axis=0)
  return batch


def make_stack_hqjitter(image, height, width, depth, BURST_LENGTH, to_shift, upscale, jitter):
  j_up = jitter * upscale
  h_up = height * upscale + 2 * j_up
  w_up = width * upscale + 2 * j_up
  v_error = tf.maximum((h_up - tf.shape(image)[0] + 1) // 2, 0)
  h_error = tf.maximum((w_up - tf.shape(image)[1] + 1) // 2, 0)
  image = tf.pad(image, [[v_error, v_error],[h_error,h_error],[0,0]])

  stack = []
  for i in range(depth):
    stack.append(tf.random_crop(image, [h_up, w_up, 3]))
  stack = tf.stack(stack, axis=0)
  return stack


def load_batch_hqjitter(dataset_dir, patches_per_img=32, min_queue=8, BURST_LENGTH=1, batch_size=32,
                        repeats=1, height=64, width=64, degamma=1.,
                        to_shift=1., upscale=1, jitter=1, smalljitter=1):

  filenames = [os.path.join(dataset_dir, f) for f in gfile.ListDirectory(dataset_dir)]
  filename_queue = tf.train.string_input_producer(filenames)

  _, image_file = tf.WholeFileReader().read(filename_queue)
  image = tf.image.decode_image(image_file)
  patches = make_stack_hqjitter((tf.cast(image[0], tf.float32) / 255.)**degamma,
                                                    height, width, patches_per_img, BURST_LENGTH, to_shift, upscale, jitter)
  unique = batch_size//repeats
  # Batch it up.
  patches  = tf.train.shuffle_batch(
        [patches],
        batch_size=unique,
        num_threads=2,
        capacity=min_queue + 3 * batch_size,
        enqueue_many=True,
        min_after_dequeue=min_queue)

  print('PATCHES =================',patches.get_shape().as_list())

  patches = make_batch_hqjitter(patches, BURST_LENGTH, batch_size, repeats, height, width, to_shift, upscale, jitter, smalljitter)
  return patches


def make_noised(image, height, width, sig_range):
  y = tf.random_uniform([1], 0, tf.shape(image)[0]-height, tf.int32)
  x = tf.random_uniform([1], 0, tf.shape(image)[1]-width, tf.int32)
  y, x = y[0], x[0]
  noised = tf.reshape(image[y:y+height, x:x+width, :], (1, height, width, 1, 3))
  denoised = noised
  sig = tf.random_uniform([1], 0, sig_range, tf.float32)
  noised = tf.clip_by_value(noised + tf.random_normal(tf.shape(noised),mean=0.,stddev=sig[0]),0.,1.)
  return noised, denoised, tf.reshape(sig, [1,1])

def make_stack_noised(image, height, width, depth, sig_range):
  v_error = tf.maximum(height - tf.shape(image)[0] + 1, 0)
  h_error = tf.maximum(width - tf.shape(image)[1] + 1, 0)
  image = tf.pad(image, [[0,v_error],[0,h_error],[0,0]])

  noised_stack, denoised_stack, sig_stack = make_noised(image, height, width, sig_range)
  for i in range(depth-1):
    noised, denoised, sig = make_noised(image, height, width, sig_range)
    noised_stack = tf.concat((noised_stack, noised), axis=0)
    denoised_stack = tf.concat((denoised_stack, denoised), axis=0)
    sig_stack = tf.concat((sig_stack, sig), axis=0)

  return noised_stack, denoised_stack, sig_stack

def load_batch_noised(depth, dataset_dir, batch_size=32, height=64, width=64, degamma=1., sig_range=20.):

  filenames = [os.path.join(dataset_dir, f) for f in gfile.ListDirectory(dataset_dir)]
  filename_queue = tf.train.string_input_producer(filenames)

  noised_stack = None
  while noised_stack == None:
    _, image_file = tf.WholeFileReader().read(filename_queue)
    image = tf.image.decode_image(image_file)
    noised_stack, denoised_stack, sig_stack = make_stack_noised((tf.cast(image[0], tf.float32) / 255.)**degamma, height, width, depth, sig_range)

  # Batch it up.
  noised, denoised, sig = tf.train.shuffle_batch(
        [noised_stack, denoised_stack, sig_stack],
        batch_size=batch_size,
        num_threads=2,
        capacity=1024 + 3 * batch_size,
        enqueue_many=True,
        min_after_dequeue=500)

  return noised, denoised, sig

def decode(tfr_features):
  burst = tf.decode_raw(tfr_features['burst_raw'], tf.float32)
  merged = tf.decode_raw(tfr_features['merge_raw'], tf.float32)
  readvar = tf.decode_raw(tfr_features['readvar'], tf.float32)
  shotfactor = tf.decode_raw(tfr_features['shotfactor'], tf.float32)
  channelgain = tf.decode_raw(tfr_features['channelgain'], tf.float32)
  blacklevels = tf.decode_raw(tfr_features['blacklevels'], tf.float32)
  depth = tf.cast(tfr_features['depth'], tf.int32) # 0
  height = tf.cast(tfr_features['height'], tf.int32) # 1
  width = tf.cast(tfr_features['width'], tf.int32) # 2
  #   depth = width_
  #   height = depth_
  #   width = height_
  #   WIDTH=4032
  #   HEIGHT=3024
  #       payload_raw_c = (payload_raw-bl/16) * ch
  burst = tf.reshape(burst, (height,width,depth))
  sh = tf.shape(burst)
  ch = tf.tile(tf.reshape(channelgain, (2,2,1)), (sh[0]/2, sh[1]/2, sh[2]))
  bl = tf.tile(tf.reshape(blacklevels, (2,2,1)), (sh[0]/2, sh[1]/2, sh[2]))
  burst = (burst - bl/16.) * ch
  merged = tf.reshape(merged, (height,width,3)) / 16.
  scale = tf.reduce_max(merged)
  burst = tf.clip_by_value(burst, 0., scale)
  scale = 1024.
  burst = burst / scale
  merged = merged / scale
  readvar = tf.reshape(readvar * channelgain * channelgain, [4]) / scale / scale
  shotfactor = tf.reshape(shotfactor * channelgain, [4]) / scale

  return burst, merged, readvar, shotfactor



def decode_patches(tfr_features):
  burst = tf.decode_raw(tfr_features['burst_raw'], tf.float32)
  merged = tf.decode_raw(tfr_features['merge_raw'], tf.float32)
  demosaic = tf.decode_raw(tfr_features['demosaic_raw'], tf.float32)
  readvar = tf.decode_raw(tfr_features['readvar'], tf.float32)
  shotfactor = tf.decode_raw(tfr_features['shotfactor'], tf.float32)
  channelgain = tf.decode_raw(tfr_features['channelgain'], tf.float32)
  blacklevels = tf.decode_raw(tfr_features['blacklevels'], tf.float32)
  depth = tf.cast(tfr_features['depth'], tf.int32) # 0
  height = tf.cast(tfr_features['height'], tf.int32) # 1
  width = tf.cast(tfr_features['width'], tf.int32) # 2
  patches = tf.cast(tfr_features['patches'], tf.int32)

  burst = tf.reshape(burst, (patches, height,width,depth))
  sh = tf.shape(burst)
  ch = tf.tile(tf.reshape(channelgain, (2,2,1)), (sh[1]/2, sh[2]/2, sh[3]))
  bl = tf.tile(tf.reshape(blacklevels, (2,2,1)), (sh[1]/2, sh[2]/2, sh[3]))
  burst = (burst - bl/16./2**10) * ch
  merged = tf.reshape(merged, (patches,height,width))
  demosaic = tf.reshape(demosaic, (patches,height,width,3))
  demosaic = demosaic

  burst = tf.clip_by_value(burst, -10, 1.)
  merged = tf.clip_by_value(merged, -10, 1.)
  scale = 2.**10
  readvar = tf.reshape(readvar, [4]) / scale / scale
  shotfactor = tf.reshape(shotfactor, [4]) / scale

  return burst, merged, demosaic, readvar, shotfactor

def read_and_decode_single(filename):
  e = tf.python_io.tf_record_iterator(filename).next()
  features = tf.parse_single_example(e, features={
          'readvar': tf.FixedLenFeature([], tf.string),
          'shotfactor': tf.FixedLenFeature([], tf.string),
          'blacklevels': tf.FixedLenFeature([], tf.string),
          'channelgain': tf.FixedLenFeature([], tf.string),
          'burst_raw': tf.FixedLenFeature([], tf.string),
          'merge_raw': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
      })

  return decode(features)

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'readvar': tf.FixedLenFeature([], tf.string),
          'shotfactor': tf.FixedLenFeature([], tf.string),
          'blacklevels': tf.FixedLenFeature([], tf.string),
          'channelgain': tf.FixedLenFeature([], tf.string),
          'burst_raw': tf.FixedLenFeature([], tf.string),
          'merge_raw': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
      })

  return decode(features)

def read_and_decode_patches(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'readvar': tf.FixedLenFeature([], tf.string),
          'shotfactor': tf.FixedLenFeature([], tf.string),
          'blacklevels': tf.FixedLenFeature([], tf.string),
          'channelgain': tf.FixedLenFeature([], tf.string),
          'burst_raw': tf.FixedLenFeature([], tf.string),
          'merge_raw': tf.FixedLenFeature([], tf.string),
          'demosaic_raw': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'patches': tf.FixedLenFeature([], tf.int64),
      })

  return decode_patches(features)

def read_and_decode_str(filename_queue):
  reader = tf.TFRecordReader()
  s, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'readvar': tf.FixedLenFeature([], tf.string),
          'shotfactor': tf.FixedLenFeature([], tf.string),
          'blacklevels': tf.FixedLenFeature([], tf.string),
          'channelgain': tf.FixedLenFeature([], tf.string),
          'burst_raw': tf.FixedLenFeature([], tf.string),
          'merge_raw': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
      })

  return s, decode(features)


def load_tfrecord(filename):
  g = tf.Graph()
  with g.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    mosaic, demosaic_truth, readvar, shotfactor = read_and_decode_single(filename)
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    with tf.Session() as sess:
      sess.run(init_op)
      mosaic, demosaic_truth, readvar, shotfactor = \
        sess.run([mosaic, demosaic_truth, readvar, shotfactor])

      return mosaic, demosaic_truth, readvar, shotfactor

def sample_patch(burst, merged, height, width, burst_length):
  y = tf.random_uniform([1], 0, tf.shape(burst)[0]-height, tf.int32)
  x = tf.random_uniform([1], 0, tf.shape(burst)[1]-width, tf.int32)
  y, x = (y[0]//2)*2, (x[0]//2)*2
  mosaic = burst[y:y+height, x:x+width,:burst_length]
  demosaic = merged[y:y+height, x:x+width,:]
  return mosaic, demosaic

def stackRGB(burst):
  burst = tf.stack((burst[:,::2,::2],(burst[:,1::2,::2]+burst[:,::2,1::2])/2,burst[:,1::2,1::2]), axis=-1)
  return burst

def burst2patches(burst, merged, height, width, depth, burst_length):

  mosaic, demosaic = sample_patch(burst, merged, height, width, burst_length)
  mosaic = tf.expand_dims(mosaic, axis=0)
  demosaic = tf.expand_dims(demosaic, axis=0)
  for i in range(depth-1):
    m, d = sample_patch(burst, merged, height, width, burst_length)
    m = tf.expand_dims(m, axis=0)
    d = tf.expand_dims(d, axis=0)
    mosaic = tf.concat((mosaic, m), axis=0)
    demosaic = tf.concat((demosaic, d), axis=0)

  return mosaic, demosaic


def inputs(filenames, batch_size, height, width, depth, burst_length):

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(filenames)

    # Even when reading in multiple threads, share the filename
    # queue.
    burst, merged, readvar, shotfactor = read_and_decode(filename_queue)

    d = tf.shape(burst)[-1]
    burst = tf.cond(d > burst_length, lambda: burst[...,:burst_length], lambda: burst)
    burst = tf.cond(d < burst_length, lambda: tf.pad(burst, [[0,0],[0,0],[0,burst_length-d]]), lambda: burst)

    mosaic, demosaic = burst2patches(burst, merged, height, width, depth, burst_length)
    mosaic = tf.reshape(mosaic, [depth, height, width, burst_length])
    demosaic = tf.reshape(demosaic, [depth, height, width, 3])
    readvar = tf.tile(tf.reshape(readvar, [1, 4]), [depth, 1])
    shotfactor = tf.tile(tf.reshape(shotfactor, [1, 4]), [depth, 1])

    valid_mask = tf.ones([1,tf.minimum(burst_length,d)])
    valid_mask = tf.cond(burst_length > d, lambda : tf.concat([valid_mask,tf.zeros([1,burst_length-d])], axis=-1), lambda : valid_mask)
    valid_mask = tf.tile(valid_mask, [depth, 1])
    valid_mask = tf.reshape(valid_mask, [depth, burst_length])

    mosaic, demosaic, readvar, shotfactor, valid_mask = tf.train.shuffle_batch(
          [mosaic, demosaic, readvar, shotfactor, valid_mask],
          batch_size=batch_size,
          num_threads=2,
          capacity=1024 + 3 * batch_size,
          enqueue_many=True,
          min_after_dequeue=128)

    return mosaic, demosaic, readvar, shotfactor, valid_mask


def inputs_patches(filenames, batch_size, burst_length):

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(filenames)

    # Even when reading in multiple threads, share the filename
    # queue.
    burst, merged, demosaic, readvar, shotfactor = read_and_decode_patches(filename_queue)

#     d = tf.shape(burst)[-1]
#     burst = tf.cond(d > burst_length, lambda: burst[...,:burst_length], lambda: burst)
#     burst = tf.cond(d < burst_length, lambda: tf.pad(burst, [[0,0],[0,0],[0,0],[0,burst_length-d]]), lambda: burst)
    burst = burst[...,:burst_length]

    depth = 16 # tf.shape(burst)[0]
    readvar = tf.tile(tf.reshape(readvar, [1, 4]), [depth, 1])
    shotfactor = tf.tile(tf.reshape(shotfactor, [1, 4]), [depth, 1])
    burst = tf.reshape(burst, [depth, 256, 256, burst_length])
    merged = tf.reshape(merged, [depth, 256, 256])
    demosaic = tf.reshape(demosaic, [depth, 256, 256, 3])

    burst, merged, demosaic, readvar, shotfactor = tf.train.shuffle_batch(
          [burst, merged, demosaic, readvar, shotfactor],
          batch_size=batch_size,
          num_threads=2,
          capacity=1000 + 3 * batch_size,
          enqueue_many=True,
          min_after_dequeue=128)

    return burst, merged, demosaic, readvar, shotfactor

def load_test_patches(filenames, burst_length):
  with tf.Graph().as_default():

    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)

      # Even when reading in multiple threads, share the filename
      # queue.
      burst, merged, demosaic, readvar, shotfactor = read_and_decode_patches(filename_queue)
      burst = burst[...,:burst_length]
      depth = 16 # tf.shape(burst)[0]
      readvar = tf.tile(tf.reshape(readvar, [1, 4]), [depth, 1])
      shotfactor = tf.tile(tf.reshape(shotfactor, [1, 4]), [depth, 1])
      burst = tf.reshape(burst, [depth, 256, 256, burst_length])
      merged = tf.reshape(merged, [depth, 256, 256])
      demosaic = tf.reshape(demosaic, [depth, 256, 256, 3])

    init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

    with tf.Session() as sess:
      sess.run(init_op)
      with queues.QueueRunners(sess):
        patches = {}

        for i,f in enumerate(filenames):
          print 'loading',f,'its', i,'of',len(filenames)
          burst_np, merged_np, demosaic_np, readvar_np, shotfactor_np = sess.run([burst, merged, demosaic, readvar, shotfactor])
          patches[f] = [burst_np, merged_np, demosaic_np, readvar_np, shotfactor_np]

        return patches
