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
import argparse
import time
import numpy as np
import cv2
import sys

from bts_dataloader import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',          type=str,   help='model name', default='bts_v0_0_1')
parser.add_argument('--encoder',             type=str,   help='type of encoder, desenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--data_path',           type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',             type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',      type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',        type=int,   help='input height', default=480)
parser.add_argument('--input_width',         type=int,   help='input width', default=640)
parser.add_argument('--max_depth',           type=float, help='maximum depth in estimation',        default=80)
parser.add_argument('--output_directory',    type=str,   help='output directory for summary, if empty outputs to checkpoint folder', default='')
parser.add_argument('--checkpoint_path',     type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset',             type=str,   help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',     action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',     action='store_true')

parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation',        default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation',        default=80)
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


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    global gt_depths, is_missing, missing_ids
    gt_depths = []
    is_missing = []
    missing_ids = set()

    write_summary = False

    if os.path.exists(args.checkpoint_path + '.meta'):
        steps = [str(args.checkpoint_path).split('/')[-1].split('-')[-1]]
    else:
        with open(args.checkpoint_path + '/checkpoint') as file:
            lines = file.readlines()[1:]

        steps = set()
        for line in lines:
            step = line.split()[1].split('/')[-1].split('-')[-1].replace('\"', '')
            steps.add('{:06d}'.format(int(step)))
    
        lines = []
        if os.path.exists(args.checkpoint_path + '/evaluated_checkpoints'):
            with open(args.checkpoint_path + '/evaluated_checkpoints') as file:
                lines = file.readlines()
    
        for line in lines:
            if line.rstrip() in steps:
                steps.remove(line.rstrip())
    
        steps = sorted(steps)
        if args.output_directory != '':
            summary_path = os.path.join(args.output_directory, args.model_name)
        else:
            summary_path = os.path.join(args.checkpoint_path, 'eval')
        write_summary = True

    if len(steps) == 0:
        print('No new model to evaluate. Abort.')
        return

    time_modified = os.path.getmtime(args.checkpoint_path + 'checkpoint')
    time_diff = time.time() - time_modified
    if time_diff < 60:
        print('Model file might not be mature due to short time_diff: %s' % str(time_diff))
        print('Aborting')
        return
    else:
        print('time_diff: %s' % str(time_diff))

    dataloader = BtsDataloader(args.data_path, args.gt_path, args.filenames_file, params, 'test',
                               do_kb_crop=args.do_kb_crop)

    dataloader_iter = dataloader.loader.make_initializable_iterator()
    iter_init_op = dataloader_iter.initializer
    image, focal = dataloader_iter.get_next()

    model = BtsModel(params, 'test', image, None, focal=focal, bn_training=False)

    if write_summary:
        summary_writer = tf.summary.FileWriter(summary_path)

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
        for step in steps:
            
            if os.path.exists(args.checkpoint_path + '.meta'):
                restore_path = args.checkpoint_path
            else:
                restore_path = os.path.join(args.checkpoint_path, 'model-' + str(int(step)))

            # RESTORE
            train_saver.restore(sess, restore_path)
        
            num_test_samples = get_num_lines(args.filenames_file)

            with open(args.filenames_file) as f:
                lines = f.readlines()

            print('now testing {} files for step {}'.format(num_test_samples, step))
            sess.run(iter_init_op)

            pred_depths = []
            
            start_time = time.time()
            for s in range(num_test_samples):
                depth = sess.run([model.depth_est])
                pred_depths.append(depth[0].squeeze())

            elapsed_time = time.time() - start_time
            print('Elapesed time: %s' % str(elapsed_time))
            print('Done.')

            if len(gt_depths) == 0:
                for t_id in range(num_test_samples):
                    gt_depth_path = os.path.join(args.gt_path, lines[t_id].split()[1])
                    depth = cv2.imread(gt_depth_path, -1)
                    if depth is None:
                        print('Missing: %s ' % gt_depth_path)
                        missing_ids.add(t_id)
                        continue

                    if args.dataset == 'nyu':
                        depth = depth.astype(np.float32) / 1000.0
                    else:
                        depth = depth.astype(np.float32) / 256.0

                    gt_depths.append(depth)

            print('Computing errors')
            silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3 = eval(pred_depths, int(step))
            
            if write_summary:
                summary = tf.Summary()
                summary.value.add(tag='silog', simple_value=silog.mean())
                summary.value.add(tag='abs_rel', simple_value=abs_rel.mean())
                summary.value.add(tag='log10', simple_value=log10.mean())
                summary.value.add(tag='sq_rel', simple_value=sq_rel.mean())
                summary.value.add(tag='rms', simple_value=rms.mean())
                summary.value.add(tag='log_rms', simple_value=log_rms.mean())
                summary.value.add(tag='d1', simple_value=d1.mean())
                summary.value.add(tag='d2', simple_value=d2.mean())
                summary.value.add(tag='d3', simple_value=d3.mean())
                
                summary_writer.add_summary(summary, global_step=step)
                summary_writer.flush()
                
                with open(os.path.dirname(args.checkpoint_path) + '/evaluated_checkpoints', 'a') as file:
                        file.write(step + '\n')
            
            print('Evaluation done')


def eval(pred_depths, step):

    num_samples = get_num_lines(args.filenames_file)
    pred_depths_valid = []

    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_ids)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths_valid[i]

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    print("{:7.4f}, {:7.4f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        silog.mean(), abs_rel.mean(), log10.mean(), rms.mean(), sq_rel.mean(), log_rms.mean(), d1.mean(), d2.mean(), d3.mean()))
    
    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


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



