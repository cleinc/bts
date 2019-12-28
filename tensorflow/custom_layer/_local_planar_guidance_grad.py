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

import tensorflow as tf
from tensorflow.python.framework import ops

lpg = tf.load_op_library('custom_layer/build/liblpg.so')

@ops.RegisterGradient("LocalPlanarGuidance")
def _local_planar_guidance_grad_cc(op, depth_grad):
    """
    The gradient for `local_planar_guidance` using the operation implemented in C++.

    :param op: `local_planar_guidance` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `local_planar_guidance` op.
    :return: gradients with respect to the input of `local_planar_guidance`.
    """
    
    return lpg.local_planar_guidance_grad(depth_grad, op.inputs[0], op.inputs[1])
