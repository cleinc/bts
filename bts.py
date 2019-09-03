# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.append("./custom_layer/")

import _compute_depth_grad

compute_depth_module = tf.load_op_library('custom_layer/build/libcompute_depth.so')

bts_parameters = namedtuple('parameters', 'encoder, '
                                          'height, width, '
                                          'max_depth, '
                                          'batch_size, '
                                          'dataset, '
                                          'num_gpus, '
                                          'num_threads, '
                                          'num_epochs, ')


class BtsModel(object):

    def __init__(self, params, mode, image, depth_gt, focal, reuse_variables=None, model_index=0, bn_training=False):
        self.params = params
        self.mode = mode
        self.max_depth = self.params.max_depth

        self.input_image = image
        self.depth_gt = depth_gt
        self.focal = tf.cast(focal, tf.float32)
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables
        self.bn_training = bn_training
        self.is_training = True if mode == 'train' else False

        self.build_model(net_input=self.input_image, reuse=self.reuse_variables)

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio], align_corners=True)

    def downsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = tf.cast(s[1] / ratio, tf.int32)
        w = tf.cast(s[2] / ratio, tf.int32)
        return tf.image.resize_nearest_neighbor(x, [h, w], align_corners=True)

    def get_depth(self, x):
        depth = self.max_depth * self.conv(x, 1, 3, 1, tf.nn.sigmoid, normalizer_fn=None)
        return depth

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu, normalizer_fn=None):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn,
                           normalizer_fn=normalizer_fn)

    def atrous_conv(self, x, num_out_layers, kernel_size, rate, apply_bn_first=True):
        pk = np.floor((kernel_size - 1) / 2).astype(np.int32)
        pr = rate - 1
        p = pk + pr
        out = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])

        if apply_bn_first is True:
            out = slim.batch_norm(out)

        out = tf.nn.relu(out)
        out = slim.conv2d(out, num_out_layers * 2, 1, 1, 'VALID')
        out = slim.batch_norm(out)
        out = tf.nn.relu(out)
        out = slim.conv2d(out, num_out_layers, kernel_size=kernel_size, stride=1, rate=rate, padding='VALID',
                          activation_fn=None, normalizer_fn=None)

        return out

    def upconv(self, x, num_out_layers, kernel_size, scale, normalizer_fn=None):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1, normalizer_fn=normalizer_fn)
        return conv

    @slim.add_arg_scope
    def denseconv(self, x, num_filters, kernel_size, stride=1, dilation_rate=1, dropout_rate=None, scope=None):
        with tf.variable_scope(scope, 'xx', [x]) as sc:
            out = slim.batch_norm(x, is_training=False)
            out = tf.nn.relu(out)
            out = slim.conv2d(out, num_filters, kernel_size, rate=dilation_rate, activation_fn=None)
            if dropout_rate:
                out = tf.nn.dropout(out)
            return out

    @slim.add_arg_scope
    def denseconv_block(self, x, num_filters, dilation_rate=1, scope=None):
        with tf.variable_scope(scope, 'conv_blockx', [x]) as sc:
            out = self.denseconv(x, num_filters * 4, 1, scope='x1')
            out = self.denseconv(out, num_filters, 3, dilation_rate=dilation_rate, scope='x2')
            out = tf.concat([x, out], axis=3)
            return out

    @slim.add_arg_scope
    def dense_block(self, x, num_layers, num_filters, growth_rate, dilation_rate=1, grow_num_filters=True, scope=None):
        with tf.variable_scope(scope, 'dense_blockx', [x]) as sc:
            out = x
            for i in range(num_layers):
                branch = i + 1
                out = self.denseconv_block(out, growth_rate, dilation_rate=dilation_rate,
                                           scope='conv_block' + str(branch))
                if grow_num_filters:
                    num_filters += growth_rate
            return out, num_filters

    @slim.add_arg_scope
    def transition_block(self, x, num_filters, compression=1.0, do_pooling=True, scope=None):
        num_filters = int(num_filters * compression)
        with tf.variable_scope(scope, 'transition_blockx', [x]) as sc:
            out = self.denseconv(x, num_filters, 1, scope='blk')
            if do_pooling:
                out = slim.avg_pool2d(out, 2)
            return out, num_filters

    @slim.add_arg_scope
    def reduction_1x1(self, net, num_filters):
        while num_filters >= 4:
            if num_filters < 8:
                num_filters = 4
                net = self.conv(net, num_filters, 1, 1, activation_fn=self.custom_sigmoid)
                break
            else:
                net = self.conv(net, num_filters, 1, 1)

            num_filters = num_filters / 2

        assert num_filters is 4
        return net

    def custom_sigmoid(self, x):
        return tf.concat([tf.nn.tanh(x[:, :, :, 0:2]), tf.expand_dims(tf.nn.sigmoid(x[:, :, :, 2]), 3),
                          tf.expand_dims(tf.nn.sigmoid(x[:, :, :, 3]) * self.max_depth, 3)], axis=3)

    def densenet(self, inputs, reduction=None, growth_rate=None, num_filters=None, num_layers=None, dropout_rate=None,
                 is_training=True, reuse=None, scope=None):

        assert reduction is not None
        assert growth_rate is not None
        assert num_filters is not None
        assert num_layers is not None

        compression = 1.0 - reduction
        num_dense_blocks = len(num_layers)

        batch_norm_params = {'is_training': False,
                             'scale': True,
                             'decay': 0.99,
                             'epsilon': 1.1e-5,
                             'fused': True, }

        with tf.variable_scope(scope, 'densenetxxx', [inputs], reuse=reuse) as sc:
            with slim.arg_scope([slim.dropout], is_training=is_training),\
                 slim.arg_scope([slim.batch_norm], **batch_norm_params),\
                 slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4), activation_fn=None, biases_initializer=None):

                skips = []

                net = inputs

                # Initial convolution
                net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')  # H/2
                net = slim.batch_norm(net, is_training=False)
                net = tf.nn.relu(net)

                skips.append(net)

                net = slim.max_pool2d(net, 3, stride=2, padding='SAME')  # H/4
                skips.append(net)

                # Blocks
                for i in range(num_dense_blocks - 1):  # i:0 H/8, i:1 H/16, i:2 H/32
                    do_pooling = True
                    dilation_rate = 1

                    net, num_filters = self.dense_block(net, num_layers[i], num_filters, growth_rate,
                                                        dilation_rate=dilation_rate, scope='dense_block' + str(i + 1))

                    # Add transition_block
                    net, num_filters = self.transition_block(net, num_filters, compression=compression,
                                                             do_pooling=do_pooling,
                                                             scope='transition_block' + str(i + 1))
                    if i < num_dense_blocks - 2:
                        skips.append(net)

                net, num_filters = self.dense_block(net, num_layers[-1], num_filters, growth_rate,
                                                    scope='dense_block' + str(num_dense_blocks))

                with tf.variable_scope('final_block', [inputs]):
                    net = slim.batch_norm(net, is_training=False)
                    net = tf.nn.relu(net)

                return net, skips

    @slim.add_arg_scope
    def bts(self, dense_features, skips, num_filters=256):
        batch_norm_params = {'is_training': self.bn_training,
                             'scale': True,
                             'decay': 0.99,
                             'epsilon': 1.1e-5,
                             'fused': True, }

        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            conv = self.conv
            atrous_conv = self.atrous_conv
            upconv = self.upconv

            upconv5 = upconv(dense_features, num_filters, 3, 2)  # H/16
            upconv5 = slim.batch_norm(upconv5)
            concat5 = tf.concat([upconv5, skips[3]], 3)
            iconv5 = conv(concat5, num_filters, 3, 1)

            num_filters = num_filters / 2

            upconv4 = upconv(iconv5, num_filters, 3, 2)  # H/8
            upconv4 = slim.batch_norm(upconv4)
            concat4 = tf.concat([upconv4, skips[2]], 3)
            iconv4 = conv(concat4, num_filters, 3, 1)
            iconv4 = slim.batch_norm(iconv4)

            daspp_3 = atrous_conv(iconv4, num_filters / 2, kernel_size=3, rate=3, apply_bn_first=False)
            concat4_2 = tf.concat([concat4, daspp_3], 3)
            daspp_6 = atrous_conv(concat4_2, num_filters / 2, kernel_size=3, rate=6)
            concat4_3 = tf.concat([concat4_2, daspp_6], 3)
            daspp_12 = atrous_conv(concat4_3, num_filters / 2, kernel_size=3, rate=12)
            concat4_4 = tf.concat([concat4_3, daspp_12], 3)
            daspp_18 = atrous_conv(concat4_4, num_filters / 2, kernel_size=3, rate=18)
            concat4_5 = tf.concat([concat4_4, daspp_18], 3)
            daspp_24 = atrous_conv(concat4_5, num_filters / 2, kernel_size=3, rate=24)
            concat4_daspp = tf.concat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], 3)
            daspp_feat = conv(concat4_daspp, num_filters / 2, 3, 1)

            rconv1_8x8 = self.reduction_1x1(daspp_feat, num_filters / 2)
            plane_normal_8x8 = tf.nn.l2_normalize(rconv1_8x8[:, :, :, 0:3], axis=3)
            plane_dist_8x8 = rconv1_8x8[:, :, :, 3]
            plane_eq_8x8 = tf.concat([plane_normal_8x8, tf.expand_dims(plane_dist_8x8, 3)], 3)
            depth_8x8 = compute_depth_module.compute_depth(plane_eq_8x8, upratio=8, focal=self.focal)
            depth_8x8_scaled = tf.expand_dims(depth_8x8, 3) / self.max_depth
            depth_8x8_scaled_ds = self.downsample_nn(depth_8x8_scaled, 4)

            num_filters = num_filters / 2

            upconv3 = upconv(daspp_feat, num_filters, 3, 2)  # H/4
            upconv3 = slim.batch_norm(upconv3)
            concat3 = tf.concat([upconv3, skips[1], depth_8x8_scaled_ds], 3)
            iconv3 = conv(concat3, num_filters, 3, 1)

            rconv1_4x4 = self.reduction_1x1(iconv3, num_filters / 2)
            plane_normal_4x4 = tf.nn.l2_normalize(rconv1_4x4[:, :, :, 0:3], axis=3)
            plane_dist_4x4 = rconv1_4x4[:, :, :, 3]
            plane_eq_4x4 = tf.concat([plane_normal_4x4, tf.expand_dims(plane_dist_4x4, 3)], 3)
            depth_4x4 = compute_depth_module.compute_depth(plane_eq_4x4, upratio=4, focal=self.focal)
            depth_4x4_scaled = tf.expand_dims(depth_4x4, 3) / self.max_depth
            depth_4x4_scaled_ds = self.downsample_nn(depth_4x4_scaled, 2)

            num_filters = num_filters / 2

            upconv2 = upconv(iconv3, num_filters, 3, 2)  # H/2
            upconv2 = slim.batch_norm(upconv2)
            concat2 = tf.concat([upconv2, skips[0], depth_4x4_scaled_ds], 3)
            iconv2 = conv(concat2, num_filters, 3, 1)

            rconv1_2x2 = self.reduction_1x1(iconv2, num_filters / 2)
            plane_normal_2x2 = tf.nn.l2_normalize(rconv1_2x2[:, :, :, 0:3], axis=3)
            plane_dist_2x2 = rconv1_2x2[:, :, :, 3]
            plane_eq_2x2 = tf.concat([plane_normal_2x2, tf.expand_dims(plane_dist_2x2, 3)], 3)
            depth_2x2 = compute_depth_module.compute_depth(plane_eq_2x2, upratio=2, focal=self.focal)
            depth_2x2_scaled = tf.expand_dims(depth_2x2, 3) / self.max_depth

            num_filters = num_filters / 2

            upconv1 = upconv(iconv2, num_filters, 3, 2)  # H
            concat1 = tf.concat([upconv1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], 3)
            iconv1 = conv(concat1, num_filters, 3, 1)

            self.depth_est = self.get_depth(iconv1)
            self.depth_2x2 = depth_2x2_scaled
            self.depth_4x4 = depth_4x4_scaled
            self.depth_8x8 = depth_8x8_scaled

            print("==================================")
            print(" upconv5 in/out: {} / {}".format(dense_features.shape[-1], upconv5.shape[-1]))
            print("  iconv5 in/out: {} / {}".format(concat5.shape[-1], iconv5.shape[-1]))
            print(" upconv4 in/out: {} / {}".format(iconv5.shape[-1], upconv4.shape[-1]))
            print("  iconv4 in/out: {} / {}".format(concat4.shape[-1], iconv4.shape[-1]))
            print("    aspp in/out: {} / {}".format(concat4_daspp.shape[-1], daspp_feat.shape[-1]))
            print("reduc8x8 in/out: {} / {}".format(daspp_feat.shape[-1], rconv1_8x8.shape[-1]))
            print("  lpg8x8 in/out: {} / {}".format(plane_eq_8x8.shape[-1], 1))
            print(" upconv3 in/out: {} / {}".format(daspp_feat.shape[-1], upconv3.shape[-1]))
            print("  iconv3 in/out: {} / {}".format(concat3.shape[-1], iconv3.shape[-1]))
            print("reduc4x4 in/out: {} / {}".format(iconv3.shape[-1], rconv1_4x4.shape[-1]))
            print("  lpg4x4 in/out: {} / {}".format(plane_eq_4x4.shape[-1], 1))
            print(" upconv2 in/out: {} / {}".format(iconv3.shape[-1], upconv2.shape[-1]))
            print("  iconv2 in/out: {} / {}".format(concat2.shape[-1], iconv2.shape[-1]))
            print("reduc2x2 in/out: {} / {}".format(iconv2.shape[-1], rconv1_2x2.shape[-1]))
            print("  lpg2x2 in/out: {} / {}".format(plane_eq_2x2.shape[-1], 1))
            print(" upconv1 in/out: {} / {}".format(iconv2.shape[-1], upconv1.shape[-1]))
            print("  iconv1 in/out: {} / {}".format(concat1.shape[-1], iconv1.shape[-1]))
            print("   depth in/out: {} / {}".format(iconv1.shape[-1], self.depth_est.shape[-1]))
            print("==================================")

    def build_densenet121_bts(self, net_input, reuse):
        with tf.variable_scope('encoder'):
            dense_features, skips = self.densenet(net_input, reduction=0.5, growth_rate=32,
                                                  num_filters=self.num_filters, num_layers=[6, 12, 24, 16],
                                                  is_training=self.is_training, reuse=reuse, scope='densenet121')

        with tf.variable_scope('decoder'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
                self.bts(dense_features, skips, num_filters=256)

    def build_densenet161_bts(self, net_input, reuse):
        with tf.variable_scope('encoder'):
            dense_features, skips = self.densenet(net_input, reduction=0.5, growth_rate=48,
                                                  num_filters=self.num_filters, num_layers=[6, 12, 36, 24],
                                                  is_training=self.is_training, reuse=reuse, scope='densenet161')

        with tf.variable_scope('decoder'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
                self.bts(dense_features, skips, num_filters=512)

    def build_model(self, net_input, reuse):
        with tf.variable_scope('model', reuse=reuse):
            if self.params.encoder == 'densenet161_bts':
                self.num_filters = 96
                self.build_densenet161_bts(net_input=net_input, reuse=reuse)
            elif self.params.encoder == 'densenet121_bts':
                self.num_filters = 64
                self.build_densenet121_bts(net_input=net_input, reuse=reuse)
            else:
                return None

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):

            if self.params.dataset == 'nyu':
                self.mask = self.depth_gt > 0.1
            else:
                self.mask = self.depth_gt > 1.0

            depth_gt_masked = tf.boolean_mask(self.depth_gt, self.mask)
            depth_est_masked = tf.boolean_mask(self.depth_est, self.mask)

            d = tf.log(depth_est_masked) - tf.log(depth_gt_masked)  # Best

            self.silog_loss = tf.sqrt(tf.reduce_mean(d ** 2) - 0.85 * (tf.reduce_mean(d) ** 2)) * 10.0
            self.total_loss = self.silog_loss

    def build_summaries(self):
        with tf.device('/cpu:0'):
            tf.summary.scalar('silog_loss', self.silog_loss, collections=self.model_collection)

            depth_gt = tf.where(self.depth_gt < 1e-3, self.depth_gt * 0 + 1e3, self.depth_gt)
            tf.summary.image('depth_gt', 1 / depth_gt, max_outputs=4, collections=self.model_collection)
            tf.summary.image('depth_est', 1 / self.depth_est, max_outputs=4, collections=self.model_collection)
            tf.summary.image('depth_est_cropped',
                             1 / self.depth_est[:, 8:self.params.height - 8, 8:self.params.width - 8, :], max_outputs=4,
                             collections=self.model_collection)
            tf.summary.image('depth_est_2x2', 1 / self.depth_2x2, max_outputs=4, collections=self.model_collection)
            tf.summary.image('depth_est_4x4', 1 / self.depth_4x4, max_outputs=4, collections=self.model_collection)
            tf.summary.image('depth_est_8x8', 1 / self.depth_8x8, max_outputs=4, collections=self.model_collection)
            tf.summary.image('image', self.input_image[:, :, :, ::-1], max_outputs=4, collections=self.model_collection)