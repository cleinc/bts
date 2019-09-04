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
import numpy as np
import argparse
import time
import tensorflow as tf
import errno
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm

from bts_dataloader import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',          type=str,   help='model name', default='bts_nyu_test')
parser.add_argument('--encoder',             type=str,   help='type of encoder, vgg or desenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--data_path',           type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',      type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',        type=int,   help='input height', default=480)
parser.add_argument('--input_width',         type=int,   help='input width', default=640)
parser.add_argument('--max_depth',           type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path',     type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset',             type=str,   help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    
    dataloader = BtsDataloader(args.data_path, None, args.filenames_file, params, 'test', do_kb_crop=args.do_kb_crop)

    dataloader_iter = dataloader.loader.make_initializable_iterator()
    iter_init_op = dataloader_iter.initializer
    image, focal = dataloader_iter.get_next()

    model = BtsModel(params, 'test', image, None, focal=focal, bn_training=False)

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

        num_test_samples = get_num_lines(args.filenames_file)

        with open(args.filenames_file) as f:
            lines = f.readlines()

        print('Now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))
        sess.run(iter_init_op)

        pred_depths = []
        pred_8x8s = []
        pred_4x4s = []
        pred_2x2s = []

        start_time = time.time()
        print('Processing images..')
        for s in tqdm(range(num_test_samples)):
            depth, pred_8x8, pred_4x4, pred_2x2 = sess.run([model.depth_est, model.depth_8x8, model.depth_4x4, model.depth_2x2])
            pred_depths.append(depth[0].squeeze())

            pred_8x8s.append(pred_8x8[0].squeeze())
            pred_4x4s.append(pred_4x4[0].squeeze())
            pred_2x2s.append(pred_2x2[0].squeeze())

        print('Done.')

        save_name = 'result_' + args.model_name

        print('Saving result pngs..')
        if not os.path.exists(os.path.dirname(save_name)):
            try:
                os.mkdir(save_name)
                os.mkdir(save_name + '/raw')
                os.mkdir(save_name + '/cmap')
                os.mkdir(save_name + '/rgb')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        for s in tqdm(range(num_test_samples)):
            if args.dataset == 'kitti':
                date_drive = lines[s].split('/')[1]
                filename_png = save_name + '/raw/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
            elif args.dataset == 'kitti_benchmark':
                filename_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
            else:
                scene_name = lines[s].split()[0].split('/')[0]
                filename_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace('.jpg', '.png')
                filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace('.jpg', '.png')
                filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]

            rgb_path = os.path.join(args.data_path, lines[s].split()[0])
            image = cv2.imread(rgb_path)
            pred_depth = pred_depths[s]
            pred_8x8 = pred_8x8s[s]
            pred_4x4 = pred_4x4s[s]
            pred_2x2 = pred_2x2s[s]

            if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
                pred_depth_scaled = pred_depth * 256.0
            else:
                pred_depth_scaled = pred_depth * 1000.0

            pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
            cv2.imwrite(filename_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            cv2.imwrite(filename_image_png, image)
            if args.dataset == 'nyu':
                pred_depth_cropped = np.zeros((480, 640), dtype=np.float32) + 1
                pred_depth_cropped[10:-1 - 10, 10:-1 - 10] = pred_depth[10:-1 - 10, 10:-1 - 10]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                pred_8x8_cropped = np.zeros((480, 640), dtype=np.float32) + 1
                pred_8x8_cropped[10:-1 - 10, 10:-1 - 10] = pred_8x8[10:-1 - 10, 10:-1 - 10]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                pred_4x4_cropped = np.zeros((480, 640), dtype=np.float32) + 1
                pred_4x4_cropped[10:-1 - 10, 10:-1 - 10] = pred_4x4[10:-1 - 10, 10:-1 - 10]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                pred_2x2_cropped = np.zeros((480, 640), dtype=np.float32) + 1
                pred_2x2_cropped[10:-1 - 10, 10:-1 - 10] = pred_2x2[10:-1 - 10, 10:-1 - 10]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')

        return


def main(_):
    
    params = bts_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=None,
        dataset=args.dataset,
        max_depth=args.max_depth,
        num_gpus=None,
        num_threads=None,
        num_epochs=None)

    test(params)


if __name__ == '__main__':
    tf.app.run()



