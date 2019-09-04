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
#include "compute_depth.h"
#include "tensorflow/core/util/cuda_kernel_helper.h" // tf <= 1.13.2
//#include "tensorflow/core/util/gpu_kernel_helper.h" // tf >= 1.14.0

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template struct functor::ComputeDepthKernel<GPUDevice>;

template struct functor::ComputeDepthGradKernel<GPUDevice>;

__global__ void ComputeDepthFunctor(const int batch_size, 
                                    const int input_height, 
                                    const int input_width,
                                    const int depth_height,
                                    const int depth_width,
                                    const float* input, 
                                    const float* focal, 
                                    float* depth)
{
    CUDA_1D_KERNEL_LOOP(index, batch_size*depth_height*depth_width)
    {
        const int num_threads_row = depth_height / input_height;
        const int num_threads_col = depth_width / input_width;

        const int col = index % depth_width;
        const int row = ((index - col) / depth_width) % depth_height;
        const int batch = ((index - row * depth_width - col) / (depth_width*depth_height)) % batch_size;

        const int input_row = row / num_threads_row;
        const int input_col = col / num_threads_col;

        float fo = focal[batch];

        float v = ((float)(row % num_threads_row) - (float)(num_threads_row - 1.0f) / 2.0f) / fo;
        float u = ((float)(col % num_threads_col) - (float)(num_threads_col - 1.0f) / 2.0f) / fo;
        unsigned int depth_index =
            batch*depth_height*depth_width + row*depth_width + col;

        unsigned int input_index =
            batch*input_height*input_width*4 + input_row*input_width*4 + input_col*4;

        float a = input[input_index+0];
        float b = input[input_index+1];
        float c = input[input_index+2];
        float d = input[input_index+3];

        float numerator = d * sqrtf(u*u + v*v + 1.0f);
        float denominator = (a*u + b*v + c);
        depth[depth_index] = numerator / denominator;
    }
}

namespace functor {
template <typename GPUDevice>
void ComputeDepthKernel<GPUDevice>::operator()(const GPUDevice& d,
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
    ComputeDepthFunctor
        <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
        (batch_size, input_height, input_width, depth_height, depth_width, input, focal, depth);
    d.synchronize();
}
}

__global__ void ComputeDepthGradFunctor(const int batch_size, 
                                        const int input_height, 
                                        const int input_width,
                                        const int depth_height,
                                        const int depth_width,
                                        const float* depth_grad, 
                                        const float* input, 
                                        const float* focal, 
                                        float* grad_input)
{
    CUDA_1D_KERNEL_LOOP(index, batch_size*input_height*input_width)
    {
        unsigned int num_threads_row = depth_height / input_height;
        unsigned int num_threads_col = depth_width / input_width;

        const int input_col = index % input_width;
        const int input_row = ((index - input_col) / input_width) % input_height;
        const int batch = ((index - input_row * input_width - input_col) / (input_width*input_height)) % batch_size;

        unsigned int input_index =
            batch*input_height*input_width*4 + input_row*input_width*4 + input_col*4;

        grad_input[input_index + 0] = 0.0f;
        grad_input[input_index + 1] = 0.0f;
        grad_input[input_index + 2] = 0.0f;
        grad_input[input_index + 3] = 0.0f;

        for(unsigned int r = 0; r < num_threads_row; ++r)
        {
            for(unsigned int cc = 0; cc < num_threads_col; ++cc)
            {
                unsigned int col = input_col * num_threads_col + cc;
                unsigned int row = input_row * num_threads_row + r;

                float fo = focal[batch];

                float v = ((float)(row % num_threads_row) - (float)(num_threads_row - 1.0f) / 2.0f) / fo;
                float u = ((float)(col % num_threads_col) - (float)(num_threads_col - 1.0f) / 2.0f) / fo;

                unsigned int depth_index =
                    batch*depth_height*depth_width + row*depth_width + col;

                float a = input[input_index + 0];
                float b = input[input_index + 1];
                float c = input[input_index + 2];
                float d = input[input_index + 3];

                float denominator = a*u + b*v + c;
                float denominator_sq = denominator*denominator;
                float numerator = -d * sqrtf(u*u + v*v + 1.0f);

		grad_input[input_index + 0] += depth_grad[depth_index] * numerator * u / denominator_sq;
		grad_input[input_index + 1] += depth_grad[depth_index] * numerator * v / denominator_sq;
		grad_input[input_index + 2] += depth_grad[depth_index] * numerator / denominator_sq;
		grad_input[input_index + 3] += depth_grad[depth_index] * sqrtf(u*u + v*v + 1.0f) / denominator;
	    }
	}
    }
}

namespace functor {
template <typename GPUDevice>
void ComputeDepthGradKernel<GPUDevice>::operator()(const GPUDevice& d,
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
    ComputeDepthGradFunctor
        <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
        (batch_size, input_height, input_width, depth_height, depth_width, depth_grad, input, focal, depth);
    d.synchronize();
}
}

#endif // GOOGLE_CUDA
