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
import tensorflow as tf
from tensorflow.python.ops import array_ops


class BtsDataloader(object):
    """bts dataloader"""

    def __init__(self, data_path, gt_path, filenames_file, params, mode,
                 do_rotate=False, degree=5.0, do_kb_crop=False):

        self.data_path = data_path
        self.gt_path = gt_path
        self.params = params
        self.mode = mode

        self.do_rotate = do_rotate
        self.degree = degree

        self.do_kb_crop = do_kb_crop

        with open(filenames_file, 'r') as f:
            filenames = f.readlines()

        if mode == 'train':
            assert not self.params.batch_size % self.params.num_gpus
            mini_batch_size = int(self.params.batch_size / self.params.num_gpus)

            self.loader = tf.data.Dataset.from_tensor_slices(filenames)
            self.loader = self.loader.apply(tf.contrib.data.shuffle_and_repeat(len(filenames)))
            self.loader = self.loader.map(self.parse_function_train, num_parallel_calls=params.num_threads)
            self.loader = self.loader.map(self.train_preprocess, num_parallel_calls=params.num_threads)
            self.loader = self.loader.batch(mini_batch_size)
            self.loader = self.loader.prefetch(mini_batch_size)

        else:
            self.loader = tf.data.Dataset.from_tensor_slices(filenames)
            self.loader = self.loader.map(self.parse_function_test, num_parallel_calls=1)
            self.loader = self.loader.map(self.test_preprocess, num_parallel_calls=1)
            self.loader = self.loader.batch(1)
            self.loader = self.loader.prefetch(1)

    def parse_function_test(self, line):
        split_line = tf.string_split([line]).values
        image_path = tf.string_join([self.data_path, split_line[0]])

        if self.params.dataset == 'nyu':
            image = tf.image.decode_jpeg(tf.read_file(image_path))
        else:
            image = tf.image.decode_png(tf.read_file(image_path))

        width_o = tf.to_float(array_ops.shape(image)[1])
        image = tf.image.convert_image_dtype(image, tf.float32)
        focal = tf.string_to_number(split_line[2])

        if self.do_kb_crop is True:
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            top_margin = tf.to_int32(height - 352)
            left_margin = tf.to_int32((width - 1216) / 2)
            image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

        return image, focal

    def test_preprocess(self, image, focal):
        # To use with model pretrained on ImageNet
        # Switch RGB to BGR order and scale to range [0,255]
        image = image[:, :, ::-1] * 255.0

        # Subtract ImageNet mean pixel values and scale
        image.set_shape([None, None, 3])
        image = self.mean_image_subtraction(image, [103.939, 116.779, 123.68]) * 0.017

        return image, focal

    def parse_function_train(self, line):
        split_line = tf.string_split([line]).values
        image_path = tf.string_join([self.data_path, split_line[0]])
        depth_gt_path = tf.string_join([self.gt_path, tf.string_strip(split_line[1])])

        if self.params.dataset == 'nyu':
            image = tf.image.decode_jpeg(tf.read_file(image_path))
        else:
            image = tf.image.decode_png(tf.read_file(image_path))

        depth_gt = tf.image.decode_png(tf.read_file(depth_gt_path), channels=0, dtype=tf.uint16)

        if self.params.dataset == 'nyu':
            depth_gt = tf.cast(depth_gt, tf.float32) / 1000.0
        else:
            depth_gt = tf.cast(depth_gt, tf.float32) / 256.0

        image = tf.image.convert_image_dtype(image, tf.float32)
        focal = tf.string_to_number(split_line[2])

        # To avoid blank boundaries due to pixel registration
        if self.params.dataset == 'nyu':
            depth_gt = depth_gt[45:472, 43:608, :]
            image = image[45:472, 43:608, :]

        if self.do_kb_crop is True:
            print('Cropping training images as kitti benchmark images')
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            top_margin = tf.to_int32(height - 352)
            left_margin = tf.to_int32((width - 1216) / 2)
            depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

        if self.do_rotate is True:
            random_angle = tf.random_uniform([], - self.degree * 3.141592 / 180, self.degree * 3.141592 / 180)
            image = tf.contrib.image.rotate(image, random_angle, interpolation='BILINEAR')
            depth_gt = tf.contrib.image.rotate(depth_gt, random_angle, interpolation='NEAREST')

        print('Do random cropping from fixed size input')
        image, depth_gt = self.random_crop_fixed_size(image, depth_gt)

        return image, depth_gt, focal

    def train_preprocess(self, image, depth_gt, focal):
        # Random flipping
        do_flip = tf.random_uniform([], 0, 1)
        image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        depth_gt = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth_gt), lambda: depth_gt)

        # Random gamma, brightness, color augmentation
        do_augment = tf.random_uniform([], 0, 1)
        image = tf.cond(do_augment > 0.5, lambda: self.augment_image(image), lambda: image)

        # To use with model pretrained on ImageNet
        # Switch RGB to BGR order and scale to range [0,255]
        image = image[:, :, ::-1] * 255.0

        image.set_shape([self.params.height, self.params.width, 3])
        depth_gt.set_shape([self.params.height, self.params.width, 1])

        # Subtract ImageNet mean pixel values and scale
        image = self.mean_image_subtraction(image, [103.939, 116.779, 123.68]) * 0.017

        return image, depth_gt, focal

    @staticmethod
    def mean_image_subtraction(image, means):
        """Subtracts the given means from each image channel.
        For example:
          means = [123.68, 116.779, 103.939]
          image = mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.
        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.
        Returns:
          the centered image.
        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
        
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
    
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def random_crop_fixed_size(self, image, depth_gt):
        image_depth = tf.concat([image, depth_gt], 2)
        image_depth_cropped = tf.random_crop(image_depth, [self.params.height, self.params.width, 4])

        image_cropped = image_depth_cropped[:, :, 0:3]
        depth_gt_cropped = tf.expand_dims(image_depth_cropped[:, :, 3], 2)

        return image_cropped, depth_gt_cropped

    @staticmethod
    def augment_image(image):
        # gamma augmentation
        gamma = tf.random_uniform([], 0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = tf.random_uniform([], 0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = tf.random_uniform([3], 0.9, 1.1)
        white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        color_image = tf.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image

        # clip
        image_aug = tf.clip_by_value(image_aug,  0, 1)

        return image_aug
