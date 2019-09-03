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

import os
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
scheduler = BlockingScheduler()

@scheduler.scheduled_job('interval', minutes=1, start_date=datetime.datetime.now() + datetime.timedelta(0,3))
def run_eval():
    command = 'export CUDA_VISIBLE_DEVICES=0; ' \
              '/usr/bin/python ' \
              'bts_eval.py ' \
              '--encoder densenet161_bts ' \
              '--dataset kitti ' \
              '--data_path ../dataset/kitti_dataset/ ' \
              '--gt_path ../dataset/kitti_dataset/data_depth_annotated/ ' \
              '--filenames_file ./train_test_inputs/eigen_test_files_with_gt.txt ' \
              '--input_height 352 ' \
              '--input_width 1216 ' \
              '--garg_crop ' \
              '--max_depth 80 ' \
              '--max_depth_eval 80 ' \
              '--output_directory ./models/eval-eigen/ ' \
              '--model_name bts_eigen_v0_0_1 ' \
              '--checkpoint_path ./models/bts_eigen_v0_0_1/ ' \
              '--do_kb_crop '

    print('Executing: %s' % command)
    os.system(command)
    print('Finished: %s' % datetime.datetime.now())

scheduler.configure()
scheduler.start()