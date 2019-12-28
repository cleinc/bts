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

#include "local_planar_guidance.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

Status LocalPlanarGuidanceShapeFn(shape_inference::InferenceContext* c)
{
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

    int upratio;
    TF_RETURN_IF_ERROR(c->GetAttr("upratio", &upratio));
    if (upratio > 1 && upratio % 2 != 0)
    {
        return errors::InvalidArgument(
                "Upratio should be multiple of 2 or 1, "
                "got: ",
                upratio);
    }
    std::vector<shape_inference::DimensionHandle> output_dims;
    for (int i = 0; i < 3; ++i) {
        shape_inference::DimensionHandle d = c->Dim(input, i);
        if (c->ValueKnown(d)) {
            int64 val;
            if(i>0)
                val = static_cast<int64>(c->Value(d) * upratio);
            else
                val = static_cast<int64>(c->Value(d));
            if (val < 0) {
                return errors::InvalidArgument("Size computed for dim ", i,
                        " is negative: ", val);
            }
            output_dims.push_back(c->MakeDim(val));
        } else {
            output_dims.push_back(c->UnknownDim());
        }
    }
    c->set_output(0, c->MakeShape(output_dims));
    return Status::OK();
}

REGISTER_OP("LocalPlanarGuidance")
.Input("input: float")
.Input("focal: float")
.Output("depth: float")
.Attr("upratio: int")
.SetShapeFn(LocalPlanarGuidanceShapeFn);

template <typename CPUDevice>
void LocalPlanarGuidanceKernel<CPUDevice>::operator()(const CPUDevice& d,
                                               const int batch_size, 
                                               const int input_height, 
                                               const int input_width,
                                               const int depth_height,
                                               const int depth_width,
                                               const float* input, 
                                               const float* focal, 
                                               float* depth)
{
    for(int index = 0; index < batch_size*depth_height*depth_width; ++index)
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

template <typename Device>
struct LocalPlanarGuidanceOp : public OpKernel {
    private:
        int upratio;
    public:
        explicit LocalPlanarGuidanceOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("upratio", &upratio));
        }

        void Compute(OpKernelContext* context) override {

            // get the input tensor
            const Tensor& input = context->input(0);
            const Tensor& focal = context->input(1);

            const TensorShape& input_shape = input.shape();

            //Check that inputs are four dimensional
            DCHECK_EQ(input_shape.dims(), 4);

            const int batch_size = input_shape.dim_size(0);
            const int input_height = input_shape.dim_size(1);
            const int input_width = input_shape.dim_size(2);
            const int norm_vec_dim_size = input_shape.dim_size(3);

            //Check
            DCHECK_EQ(norm_vec_dim_size, 4);

            const int depth_height = input_height * upratio;
            const int depth_width = input_width * upratio;

            TensorShape depth_shape;
            depth_shape.AddDim(batch_size);
            depth_shape.AddDim(depth_height);
            depth_shape.AddDim(depth_width);
            Tensor* depth = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, depth_shape, &depth));

            LocalPlanarGuidanceKernel<Device>()(context->eigen_device<Device>(),
                                         batch_size,
                                         input_height,
                                         input_width,
                                         depth_height,
                                         depth_width,
                                         input.flat<float>().data(),
                                         focal.flat<float>().data(),
                                         depth->flat<float>().data());
        }
};
REGISTER_KERNEL_BUILDER(Name("LocalPlanarGuidance").Device(DEVICE_CPU), LocalPlanarGuidanceOp<CPUDevice>);
#ifdef GOOGLE_CUDA
namespace functor {
template <>                                                          
void LocalPlanarGuidanceKernel<GPUDevice>::operator()(const GPUDevice& d,
                                               const int batch_size, 
                                               const int input_height, 
                                               const int input_width,
                                               const int depth_height,
                                               const int depth_width,
                                               const float* input, 
                                               const float* focal, 
                                               float* depth);
extern template struct LocalPlanarGuidanceKernel<GPUDevice>;
}
template <>
struct LocalPlanarGuidanceOp<GPUDevice> : public OpKernel {
    private:
        int upratio;
    public:
        explicit LocalPlanarGuidanceOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("upratio", &upratio));
        }

        void Compute(OpKernelContext* context) override {

            // get the input tensor
            const Tensor& input = context->input(0);
            const Tensor& focal = context->input(1);

            const TensorShape& input_shape = input.shape();

            //Check that inputs are four dimensional
            DCHECK_EQ(input_shape.dims(), 4);

            const int batch_size = input_shape.dim_size(0);
            const int input_height = input_shape.dim_size(1);
            const int input_width = input_shape.dim_size(2);
            const int norm_vec_dim_size = input_shape.dim_size(3);

            //Check
            DCHECK_EQ(norm_vec_dim_size, 4);

            const int depth_height = input_height * upratio;
            const int depth_width = input_width * upratio;

            TensorShape depth_shape;
            depth_shape.AddDim(batch_size);
            depth_shape.AddDim(depth_height);
            depth_shape.AddDim(depth_width);

            Tensor* depth = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, depth_shape, &depth));

            functor::LocalPlanarGuidanceKernel<GPUDevice>()(context->eigen_device<GPUDevice>(),
                                                     batch_size,
                                                     input_height,
                                                     input_width,
                                                     depth_height,
                                                     depth_width,
                                                     input.flat<float>().data(),
                                                     focal.flat<float>().data(),
                                                     depth->flat<float>().data());
        }
};
REGISTER_KERNEL_BUILDER(Name("LocalPlanarGuidance").Device(DEVICE_GPU), LocalPlanarGuidanceOp<GPUDevice>);
#endif

REGISTER_OP("LocalPlanarGuidanceGrad")
.Input("depth_grad: float")
.Input("input: float")
.Input("focal: float")
.Output("grad_input: float")
.Output("grad_focal: float");

template <typename CPUDevice>
void LocalPlanarGuidanceGradKernel<CPUDevice>::operator() (const CPUDevice &d,
                                                    const int batch_size, 
                                                    const int input_height, 
                                                    const int input_width,
                                                    const int depth_height,
                                                    const int depth_width,
                                                    const float* depth_grad, 
                                                    const float* input, 
                                                    const float* focal, 
                                                    float* grad_input)
{
    for(int index = 0; index < batch_size*input_height*input_width; ++index)
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

template <typename Device>
class LocalPlanarGuidanceGradOp : public OpKernel {
    public:
        explicit LocalPlanarGuidanceGradOp(OpKernelConstruction* context) : OpKernel(context) {
        }

        void Compute(OpKernelContext* context) override {
            DCHECK_EQ(3, context->num_inputs());

            const Tensor& depth_grad = context->input(0);
            const Tensor& input = context->input(1);
            const Tensor& focal = context->input(2);

            TensorShape depth_grad_shape = depth_grad.shape();
            const int depth_grad_batch_size = depth_grad_shape.dim_size(0);

            const int depth_grad_height = depth_grad_shape.dim_size(1);

            const int depth_grad_width = depth_grad_shape.dim_size(2);

            TensorShape input_shape = input.shape();

            // create output tensor
            Tensor* grad_input = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

            Tensor* grad_focal = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, focal.shape(), &grad_focal));
            // *grad_focal = focal;
            grad_focal = NULL;

            const int batch_size = input_shape.dim_size(0);
            const int input_height = input_shape.dim_size(1);
            const int input_width = input_shape.dim_size(2);

            LocalPlanarGuidanceGradKernel<Device>()(context->eigen_device<Device>(),
                                             batch_size,
                                             input_height,
                                             input_width,
                                             depth_grad_height,
                                             depth_grad_width,
                                             depth_grad.flat<float>().data(),
                                             input.flat<float>().data(),
                                             focal.flat<float>().data(),
                                             grad_input->flat<float>().data());
        }
};
REGISTER_KERNEL_BUILDER(Name("LocalPlanarGuidanceGrad").Device(DEVICE_CPU), LocalPlanarGuidanceGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
namespace functor {
template <>                                                          
void LocalPlanarGuidanceGradKernel<GPUDevice>::operator()(const GPUDevice& d,
                                                   const int batch_size, 
                                                   const int input_height, 
                                                   const int input_width,
                                                   const int depth_height,
                                                   const int depth_width,
                                                   const float* depth_grad,
                                                   const float* input, 
                                                   const float* focal, 
                                                   float* depth);
extern template struct LocalPlanarGuidanceGradKernel<GPUDevice>;
}
template<>
class LocalPlanarGuidanceGradOp<GPUDevice> : public OpKernel {
    public:
        explicit LocalPlanarGuidanceGradOp(OpKernelConstruction* context) : OpKernel(context) {
        }

        void Compute(OpKernelContext* context) override {

            DCHECK_EQ(3, context->num_inputs());

            const Tensor& depth_grad = context->input(0);
            const Tensor& input = context->input(1);
            const Tensor& focal = context->input(2);

            TensorShape depth_grad_shape = depth_grad.shape();
            const int depth_grad_batch_size = depth_grad_shape.dim_size(0);

            const int depth_grad_height = depth_grad_shape.dim_size(1);

            const int depth_grad_width = depth_grad_shape.dim_size(2);

            TensorShape input_shape = input.shape();

            // create output tensor
            Tensor* grad_input = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

            Tensor* grad_focal = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(1, focal.shape(), &grad_focal));
            *grad_focal = focal;

            // get the Eigen tensors for data access
            auto depth_grad_tensor = depth_grad.tensor<float, 3>();
            auto input_tensor = input.tensor<float, 4>();
            auto grad_input_tensor = grad_input->tensor<float, 4>();

            const int batch_size = input_shape.dim_size(0);
            const int input_height = input_shape.dim_size(1);
            const int input_width = input_shape.dim_size(2);

            functor::LocalPlanarGuidanceGradKernel<GPUDevice>()(context->eigen_device<GPUDevice>(),
                                                         batch_size,
                                                         input_height,
                                                         input_width,
                                                         depth_grad_height,
                                                         depth_grad_width,
                                                         depth_grad.flat<float>().data(),
                                                         input.flat<float>().data(),
                                                         focal.flat<float>().data(),
                                                         grad_input->flat<float>().data());
        }
};
REGISTER_KERNEL_BUILDER(Name("LocalPlanarGuidanceGrad").Device(DEVICE_GPU), LocalPlanarGuidanceGradOp<GPUDevice>);
#endif
