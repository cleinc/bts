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
import argparse
import time
import datetime
import sys

from average_gradients import *
from tensorflow.python import pywrap_tensorflow
from bts_dataloader import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=False)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=5.0)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--pretrained_model',          type=str,   help='path to a pretrained model checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
    varlist=[]
    var_value =[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)


def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    var_check = set()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
        except:
            print(tensor_name + ' is in pretrained model but not in current training model')
        if tensor_aux not in var_check:
            full_var_list.append(tensor_aux)
            var_check.add(tensor_aux)
    return full_var_list


def train(params):

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        num_training_samples = get_num_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else start_learning_rate * 0.1
        learning_rate = tf.train.polynomial_decay(start_learning_rate, global_step, num_total_steps, end_learning_rate, 0.9)
    
        opt_step = tf.train.AdamOptimizer(learning_rate, epsilon=1e-3)

        print("Total number of samples: {}".format(num_training_samples))
        print("Total number of steps: {}".format(num_total_steps))

        if args.fix_first_conv_blocks or args.fix_first_conv_block:
            if args.fix_first_conv_blocks:
                print('Fixing first two conv blocks')
            else:
                print('Fixing first conv block')

        dataloader = BtsDataloader(args.data_path, args.gt_path, args.filenames_file, params, args.mode,
                                   do_rotate=args.do_random_rotate, degree=args.degree,
                                   do_kb_crop=args.do_kb_crop)

        dataloader_iter = dataloader.loader.make_initializable_iterator()
        iter_init_op = dataloader_iter.initializer

        tower_grads = []
        tower_losses = []
        reuse_variables = None
        
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    image, depth_gt, focal = dataloader_iter.get_next()
                    model = BtsModel(params, args.mode, image, depth_gt, focal=focal,
                                     reuse_variables=reuse_variables, model_index=i, bn_training=False)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True
                    
                    if args.fix_first_conv_blocks or args.fix_first_conv_block:
                        trainable_vars = tf.trainable_variables()
                        if args.fix_first_conv_blocks:
                            g_vars = [var for var in
                                      trainable_vars  if ('conv1' or 'dense_block1' or 'dense_block2' or 'transition_block1' or 'transition_block2') not in var.name]
                        else:
                            g_vars = [var for var in
                                      trainable_vars if ('dense_block1' or 'transition_block1') not in var.name]
                    else:
                        g_vars = None
                    
                    grads = opt_step.compute_gradients(loss, var_list=g_vars)

                    tower_grads.append(grads)

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % (args.num_gpus - 1)):
                grads = average_gradients(tower_grads)
                apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)
                total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver(max_to_keep=200)

        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            
        print("Total number of trainable parameters: {}".format(total_num_parameters))
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        
        if args.pretrained_model != '':
            vars_to_restore = get_tensors_in_checkpoint_file(file_name=args.pretrained_model)
            tensors_to_load = build_tensors_in_checkpoint_file(vars_to_restore)
            loader = tf.train.Saver(tensors_to_load)
            loader.restore(sess, args.pretrained_model)

        # Load checkpoint if set
        if args.checkpoint_path != '':
            restore_path = args.checkpoint_path
            train_saver.restore(sess, restore_path)

        if args.retrain:
            sess.run(global_step.assign(0))

        start_step = global_step.eval(session=sess)
        start_time = time.time()
        duration = 0
        should_init_iter_op = False
        if args.mode == 'train':
            should_init_iter_op = True
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            if step % steps_per_epoch == 0 or should_init_iter_op is True:
                sess.run(iter_init_op)
                should_init_iter_op = False
            
            _, lr, loss_value = sess.run([apply_gradient_op, learning_rate, total_loss])
            
            print('step: {}/{}, lr: {:.12f}, loss: {:.12f}'.format(step, num_total_steps, lr, loss_value))

            duration += time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration * 100
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print('%s:' % args.model_name)
                print_string = 'examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
                summary_writer.flush()
                
            if step and step % 500 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)
        print('%s training finished' % args.model_name)
        print(datetime.datetime.now())


def main(_):
    
    params = bts_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        dataset=args.dataset,
        max_depth=args.max_depth,
        num_gpus=args.num_gpus,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs)

    if args.mode == 'train':
        model_filename = args.model_name + '.py'
        command = 'mkdir ' + args.log_directory + '/' + args.model_name
        os.system(command)

        args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
        command = 'cp ' + sys.argv[1] + ' ' + args_out_path
        os.system(command)

        if args.checkpoint_path == '':
            model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
            command = 'cp bts.py ' + model_out_path
            os.system(command)
        else:
            loaded_model_dir = os.path.dirname(args.checkpoint_path)
            loaded_model_name = os.path.basename(loaded_model_dir)
            loaded_model_filename = loaded_model_name + '.py'

            model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
            command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
            os.system(command)
        
        train(params)
        
    elif args.mode == 'test':
        print('This script does not support testing. Use bts_test.py instead.')


if __name__ == '__main__':
    tf.app.run()
