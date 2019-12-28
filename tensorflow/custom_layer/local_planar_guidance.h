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

#ifndef COMPUTE_DEPTH_H_
#define COMPUTE_DEPTH_H_

template <typename Device>
struct LocalPlanarGuidanceKernel
{
    void operator()(const Device& d,
                    const int batch_size, 
                    const int input_height, 
                    const int input_width,
                    const int depth_height,
                    const int depth_width,
                    const float* input,
                    const float* focal,
                    float* depth);
};

template <typename Device>
struct LocalPlanarGuidanceGradKernel
{
    void operator()(const Device& d,
                    const int batch_size, 
                    const int input_height, 
                    const int input_width,
                    const int depth_height,
                    const int depth_width,
                    const float* depth_grad,
                    const float* input,
                    const float* focal,
                    float* grad_input);
};

#if GOOGLE_CUDA
namespace functor { // Trick for GPU implementation forward decralation
template <typename Device>
struct LocalPlanarGuidanceKernel
{
    void operator()(const Device& d,
                    const int batch_size, 
                    const int input_height, 
                    const int input_width,
                    const int depth_height,
                    const int depth_width,
                    const float* input,
                    const float* focal,
                    float* depth);
};

template <typename Device>
struct LocalPlanarGuidanceGradKernel
{
    void operator()(const Device& d,
                    const int batch_size, 
                    const int input_height, 
                    const int input_width,
                    const int depth_height,
                    const int depth_width,
                    const float* depth_grad,
                    const float* input,
                    const float* focal,
                    float* grad_input);
};
}
#endif

#endif // COMPUTE_DEPTH_H_