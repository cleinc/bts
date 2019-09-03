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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import argparse
import time
import glob
import cv2
import errno
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

from bts_dataloader import *

parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.')

parser.add_argument('--model_name',      type=str,   help='model name', default='bts_v0_0_1')
parser.add_argument('--encoder',         type=str,   help='type of encoder, densenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--dataset',         type=str,   help='dataset to test, kitti or nyu', default='')
parser.add_argument('--max_depth',       type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--focal',           type=float, help='focal length in pixels', default=-1)
parser.add_argument('--image_path',      type=str,   help='image sequence path', required=True)
parser.add_argument('--out_path',        type=str,   help='output path', required=True)
parser.add_argument('--checkpoint_path', type=str,   help='path to a checkpoint to load', required=True)
parser.add_argument('--input_height',    type=int,   help='input height', default=480)
parser.add_argument('--input_width',     type=int,   help='input width',  default=640)

args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)

sys.path.append(model_dir)
for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def test_sequence(params):
    image_files = []

    for filename in glob.glob(os.path.join(args.image_path, '*.png')):
        image_files.append(filename)

    image_files.sort()

    num_test_samples = len(image_files)
    if num_test_samples == 0:
        print("No images found! Program abort.")
        return

    if args.dataset == 'nyu':
        focal = 518.8579
    elif args.dataset == 'kitti':
        focal = 718.856 # Visualize purpose only
    elif args.dataset == '' and args.focal == -1:
        print('Custom dataset needs to specify focal length with --focal')
        return

    image = tf.placeholder(tf.float32, [1, args.input_height, args.input_width, 3])
    focals = tf.constant([focal])

    model = BtsModel(params, 'test', image, None, focal=focals, bn_training=False)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # SAVER
    train_saver = tf.train.Saver()

    with tf.device('/cpu:0'):
        restore_path = args.checkpoint_path

        # RESTORE
        train_saver.restore(sess, restore_path)

        print('now testing {} files for model {}'.format(num_test_samples, args.checkpoint_path))

        print('Saving result pngs')
        if not os.path.exists(os.path.dirname(args.out_path)):
            try:
                os.mkdir(args.out_path)
                os.mkdir(args.out_path + '/depth')
                os.mkdir(args.out_path + '/lpg2x2')
                os.mkdir(args.out_path + '/lpg4x4')
                os.mkdir(args.out_path + '/lpg8x8')
                os.mkdir(args.out_path + '/rgb')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        start_time = time.time()
        for s in range(num_test_samples):
            input_image = cv2.imread(image_files[s])

            if args.dataset == 'kitti':
                height, width, ch = input_image.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                input_image = input_image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            input_image_original = input_image
            input_image = input_image.astype(np.float32)

            # Normalize image
            input_image[:, :, 0] = (input_image[:, :, 0] - 103.939) * 0.017
            input_image[:, :, 1] = (input_image[:, :, 1] - 116.779) * 0.017
            input_image[:, :, 2] = (input_image[:, :, 2] - 123.68) * 0.017

            input_images = np.reshape(input_image, (1, args.input_height, args.input_width, 3))

            depth, pred_8x8, pred_4x4, pred_2x2 = sess.run(
                [model.depth_est, model.depth_8x8, model.depth_4x4, model.depth_2x2], feed_dict={image: input_images})

            pred_depth = depth.squeeze()
            pred_8x8 = pred_8x8.squeeze()
            pred_4x4 = pred_4x4.squeeze()
            pred_2x2 = pred_2x2.squeeze()

            save_path = os.path.join(args.out_path, 'depth', image_files[s].split('/')[-1])
            plt.imsave(save_path, np.log10(pred_depth), cmap='Greys')

            save_path = os.path.join(args.out_path, 'rgb', image_files[s].split('/')[-1])
            cv2.imwrite(save_path, input_image_original)

            save_path = os.path.join(args.out_path, 'lpg2x2', image_files[s].split('/')[-1])
            plt.imsave(save_path, np.log10(pred_2x2), cmap='Greys')

            save_path = os.path.join(args.out_path, 'lpg4x4', image_files[s].split('/')[-1])
            plt.imsave(save_path, np.log10(pred_4x4), cmap='Greys')

            save_path = os.path.join(args.out_path, 'lpg8x8', image_files[s].split('/')[-1])
            plt.imsave(save_path, np.log10(pred_8x8), cmap='Greys')

            print('{}/{}'.format(s, num_test_samples))

        elapsed_time = time.time() - start_time
        print('Elapesed time: %s' % str(elapsed_time))
        print('done.')

def main(_):
    
    params = bts_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=None,
        dataset=None,
        max_depth=args.max_depth,
        num_gpus=None,
        num_threads=None,
        num_epochs=None,
        )

    test_sequence(params)

if __name__ == '__main__':
    tf.app.run()
