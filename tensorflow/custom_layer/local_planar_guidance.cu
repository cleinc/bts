/**********************************************************************
 Copyright (C) 2019 Jin Han Lee

 This file is a part of BTS.
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>
***********************************************************************/

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "local_planar_guidance.h"
#include "tensorflow/core/util/cuda_kernel_helper.h" // tf <= 1.13.2
// #include "tensorflow/core/util/gpu_kernel_helper.h" // tf >= 1.14.0

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template struct functor::LocalPlanarGuidanceKernel<GPUDevice>;

template struct functor::LocalPlanarGuidanceGradKernel<GPUDevice>;

__global__ void LocalPlanarGuidanceFunctor(const int nthreads, 
                                    const int input_height, 
                                    const int input_width,
                                    const int depth_height,
                                    const int depth_width,
                                    const float* input, 
                                    const float* focal, 
                                    float* depth)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        const int num_threads_row = depth_height / input_height;
        const int num_threads_col = depth_width / input_width;

        int batch = index;
        const int col = (batch % depth_width);
        batch /= depth_width;
        const int row = (batch % depth_height);
        batch /= depth_height;

        const int input_row = row / num_threads_row;
        const int input_col = col / num_threads_col;

        float fo = focal[batch];

        float v = ((float)(row % num_threads_row) - (float)(num_threads_row - 1.0f) / 2.0f) / (float)num_threads_row;
        float u = ((float)(col % num_threads_col) - (float)(num_threads_col - 1.0f) / 2.0f) / (float)num_threads_col;

        unsigned int input_index = batch*input_height*input_width*4 + input_row*input_width*4 + input_col*4;

        float n1 = input[input_index+0];
        float n2 = input[input_index+1];
        float n3 = input[input_index+2];
        float n4 = input[input_index+3];

        float numerator = n4;
        float denominator = (n1*u + n2*v + n3);
        depth[index] = numerator / denominator;
    }
}

namespace functor {
template <typename GPUDevice>
void LocalPlanarGuidanceKernel<GPUDevice>::operator()(const GPUDevice& d,
                                               const int batch_size, 
                                               const int input_height, 
                                               const int input_width,
                                               const int depth_height,
                                               const int depth_width,
                                               const float* input, 
                                               const float* focal, 
                                               float* depth)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size*depth_height*depth_width;
    LocalPlanarGuidanceFunctor
        <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
        (output_size, input_height, input_width, depth_height, depth_width, input, focal, depth);
    d.synchronize();
}
}

__global__ void LocalPlanarGuidanceGradFunctor(const int nthreads, 
                                        const int input_height, 
                                        const int input_width,
                                        const int depth_height,
                                        const int depth_width,
                                        const float* depth_grad, 
                                        const float* input, 
                                        const float* focal, 
                                        float* grad_input)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        unsigned int num_threads_row = depth_height / input_height;
        unsigned int num_threads_col = depth_width / input_width;

        int batch = index;
        const int input_col = (batch % input_width);
        batch /= input_width;
        const int input_row = (batch % input_height);
        batch /= input_height;

        grad_input[index * 4 + 0] = 0.0f;
        grad_input[index * 4 + 1] = 0.0f;
        grad_input[index * 4 + 2] = 0.0f;
        grad_input[index * 4 + 3] = 0.0f;

        float n1 = input[index * 4 + 0];
        float n2 = input[index * 4 + 1];
        float n3 = input[index * 4 + 2];
        float n4 = input[index * 4 + 3];

        float fo = focal[batch];

        for(unsigned int r = 0; r < num_threads_row; ++r)
        {
            for(unsigned int c = 0; c < num_threads_col; ++c)
            {
                unsigned int col = input_col * num_threads_col + c;
                unsigned int row = input_row * num_threads_row + r;

                float v = ((float)(row % num_threads_row) - (float)(num_threads_row - 1.0f) / 2.0f) / (float)num_threads_row;
                float u = ((float)(col % num_threads_col) - (float)(num_threads_col - 1.0f) / 2.0f) / (float)num_threads_col;

                unsigned int depth_index = batch*depth_height*depth_width + row*depth_width + col;

                float denominator = n1*u + n2*v + n3;
                float denominator_sq = denominator*denominator;

                grad_input[index * 4 + 0] += depth_grad[depth_index] * (-1.0f * u) / denominator_sq;
                grad_input[index * 4 + 1] += depth_grad[depth_index] * (-1.0f * v) / denominator_sq;
                grad_input[index * 4 + 2] += depth_grad[depth_index] * (-1.0f) / denominator_sq;
                grad_input[index * 4 + 3] += depth_grad[depth_index] / denominator;
            }
        }
    }
}

namespace functor {
template <typename GPUDevice>
void LocalPlanarGuidanceGradKernel<GPUDevice>::operator()(const GPUDevice& d,
                                                   const int batch_size, 
                                                   const int input_height, 
                                                   const int input_width,
                                                   const int depth_height,
                                                   const int depth_width,
                                                   const float* depth_grad, 
                                                   const float* input, 
                                                   const float* focal, 
                                                   float* depth)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size*input_height*input_width;
    LocalPlanarGuidanceGradFunctor
        <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
        (output_size, input_height, input_width, depth_height, depth_width, depth_grad, input, focal, depth);
    d.synchronize();
}
}

#endif // GOOGLE_CUDA
