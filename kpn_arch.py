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
"""Contains the Mnist model definition.

The Mnist model in this file is a simple convolutional network with two
convolutional layers, two pooling layers, followed by two fully connected
layers. A single dropout layer is used between the two fully connected layers.
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tf_image import *


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


def convolve_subset(inputs, ch, N, D=3):
  inputs0 = inputs
  print 'Downsample'
  inputs = batch_down2(inputs)
  for d in range(D):
    print 'Pre-Layer with {} channels at N={}'.format(ch, N)
    inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)

  if N > 0:
    ch1 = ch*2 if ch < 512 else ch
    print 'Recursing to ch={} and N={}'.format(ch1, N-1)
    inputs, core = convolve_subset(inputs, ch1, N-1)
    for d in range(D):
      print 'Post-Layer with {} channels at N={}'.format(ch, N)
      inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)
  else:
    core = inputs

  print 'Upsample and Skip'
  inputs = tf.image.resize_images(
      inputs,
      [tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2],
      method=tf.image.ResizeMethod.BILINEAR)
  inputs = tf.concat([inputs, inputs0], axis=-1)
  return inputs, core





def convolve_net2(input_stack, conv_stack, final_K, final_W, ch0=64, N=4, D=3, scope='cnet2',
                  equiv=False, separable=False, bonus=False, avg_spatial=False):
  with tf.variable_scope(scope):
    initial_W = conv_stack.get_shape().as_list()[-1]
    inputs = input_stack
    if not separable:
      ch_final = final_K ** 2 * initial_W * final_W
    else:
      ch_final = final_K * 2 * initial_W * final_W
    # ch = 2**(10-N)
    ch = ch0
    for d in range(D):
      print 'Pre-Layer with {} channels at N={}'.format(ch, N)
      inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)

    inputs_at_0 = inputs

    for i in range(1):
      print 'Downsample'
      inputs = batch_down2(inputs)
      ch = ch * 2
      N = N-1
      for d in range(D):
        print 'Pre-Layer with {} channels at N={}'.format(ch, N)
        inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)

    inputs, core = convolve_subset(inputs, ch=ch*2, N=N-1, D=D)

    if avg_spatial:
      core = tf.layers.conv2d(core, ch_final, 1, padding='same', activation=None)
      core = tf.image.resize_images(
          core,
          [tf.shape(conv_stack)[1], tf.shape(conv_stack)[2]],
          method=tf.image.ResizeMethod.BILINEAR)
      if not separable:
        core_filts = tf.reshape(core, [tf.shape(core)[0], tf.shape(core)[1], tf.shape(core)[2], final_K, final_K, initial_W, final_W])
        core_net = convolve(conv_stack, core_filts, final_K, final_W)
      else:
        core_filts1 = tf.reshape(core[...,:final_K * initial_W * final_W], [tf.shape(core)[0], tf.shape(core)[1], tf.shape(core)[2], final_K, 1, initial_W, final_W])
        core_filts2 = tf.reshape(core[...,final_K * initial_W * final_W:], [tf.shape(core)[0], tf.shape(core)[1], tf.shape(core)[2], 1, final_K, initial_W, final_W])
        core_filts = core_filts1 * core_filts2
        core_net = convolve_aniso(conv_stack, core_filts1, final_K, 1, final_W, layerwise=True)
        core_net = convolve_aniso(core_net,   core_filts2, 1, final_K, final_W, layerwise=True)



    N = N+1
    ch = ch_final
    for d in range(2):
      print 'Post-Layer with {} channels at N={}'.format(ch, N)
      inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)

    if not equiv:
      ch = ch_final
      print 'Final-Layer with {} channels at N={}'.format(ch, N)
      inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=None)


      if False:
        inputs = tf.nn.relu(inputs)
        print 'Upsample'
        inputs = tf.image.resize_images(
            inputs,
            [tf.shape(conv_stack)[1], tf.shape(conv_stack)[2]],
            method=tf.image.ResizeMethod.BILINEAR)
        N = N+1
        inputs = tf.concat([inputs, inputs_at_0], axis=-1)
        for d in range(2):
          print 'bonus/Post-Layer with {} channels at N={}'.format(ch, N)
          inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)

        print 'bonus/Final-Layer with {} channels at N={}'.format(ch, N)
        inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=None)


      else:
        print 'Upsample'
        inputs = tf.image.resize_images(
            inputs,
            [tf.shape(conv_stack)[1], tf.shape(conv_stack)[2]],
            method=tf.image.ResizeMethod.BILINEAR)


      net = inputs
      if not separable:
        filts = tf.reshape(net, [tf.shape(net)[0], tf.shape(net)[1], tf.shape(net)[2], final_K, final_K, initial_W, final_W])
        img_net = convolve(conv_stack, filts, final_K, final_W)
      else:
        filts1 = tf.reshape(net[...,:final_K * initial_W * final_W], [tf.shape(net)[0], tf.shape(net)[1], tf.shape(net)[2], final_K, 1, initial_W, final_W])
        filts2 = tf.reshape(net[...,final_K * initial_W * final_W:], [tf.shape(net)[0], tf.shape(net)[1], tf.shape(net)[2], 1, final_K, initial_W, final_W])
        filts = filts1 * filts2
        img_net = convolve_aniso(conv_stack, filts1, final_K, 1, final_W, layerwise=True)
        img_net = convolve_aniso(img_net,    filts2, 1, final_K, final_W, layerwise=False)

      print 'Adaptive convolution applied'

      if bonus:
        inputs = img_net
        ch = ch_final
        for d in range(2):
          print 'bonus/Post-Layer with {} channels at N={}'.format(ch, N)
          inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)
        ch = final_W
        print 'bonus/Final-Layer with {} channels at N={}'.format(ch, N)
        inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=None)
        img_net = inputs

      if avg_spatial:
        return img_net, filts, core_net, core_filts
      else:
        return img_net, filts

    else:
      print 'Post-Layer with {} channels at N={}'.format(ch, N)
      inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)
      print 'Upsample'
      inputs = tf.image.resize_images(
          inputs,
          [tf.shape(conv_stack)[1], tf.shape(conv_stack)[2]],
          method=tf.image.ResizeMethod.BILINEAR)
      inputs = tf.concat([inputs, input_stack], axis=-1)
      N = N+1
      ch = ch // 4
      for d in range(2):
        print 'Post-Layer with {} channels at N={}'.format(ch, N)
        inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=tf.nn.relu)

      ch = final_W
      print 'Final-Layer with {} channels at N={}'.format(ch, N)
      inputs = tf.layers.conv2d(inputs, ch, 3, padding='same', activation=None)
      return inputs


def convolve_per_layer(conv_stack, filts, final_K, final_W):
  initial_W = conv_stack.get_shape().as_list()[-1]
  img_net = []
  for i in range(initial_W):
    img_net.append(convolve(conv_stack[...,i:i+1], filts[...,i:i+1,:], final_K, final_W))
  img_net = tf.concat(img_net, axis=-1)
  return img_net





