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

#include "compute_depth.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

Status ComputeDepthShapeFn(shape_inference::InferenceContext* c)
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

REGISTER_OP("ComputeDepth")
.Input("input: float")
.Input("focal: float")
.Output("depth: float")
.Attr("upratio: int")
.SetShapeFn(ComputeDepthShapeFn);

template <typename CPUDevice>
void ComputeDepthKernel<CPUDevice>::operator()(const CPUDevice& d,
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

template <typename Device>
struct ComputeDepthOp : public OpKernel {
    private:
        int upratio;
    public:
        explicit ComputeDepthOp(OpKernelConstruction* context) : OpKernel(context) {
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

            ComputeDepthKernel<Device>()(context->eigen_device<Device>(),
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
REGISTER_KERNEL_BUILDER(Name("ComputeDepth").Device(DEVICE_CPU), ComputeDepthOp<CPUDevice>);
#ifdef GOOGLE_CUDA
namespace functor {
template <>                                                          
void ComputeDepthKernel<GPUDevice>::operator()(const GPUDevice& d,
                                               const int batch_size, 
                                               const int input_height, 
                                               const int input_width,
                                               const int depth_height,
                                               const int depth_width,
                                               const float* input, 
                                               const float* focal, 
                                               float* depth);
extern template struct ComputeDepthKernel<GPUDevice>;
}
template <>
struct ComputeDepthOp<GPUDevice> : public OpKernel {
    private:
        int upratio;
    public:
        explicit ComputeDepthOp(OpKernelConstruction* context) : OpKernel(context) {
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

            functor::ComputeDepthKernel<GPUDevice>()(context->eigen_device<GPUDevice>(),
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
REGISTER_KERNEL_BUILDER(Name("ComputeDepth").Device(DEVICE_GPU), ComputeDepthOp<GPUDevice>);
#endif

REGISTER_OP("ComputeDepthGrad")
.Input("depth_grad: float")
.Input("input: float")
.Input("focal: float")
.Output("grad_input: float")
.Output("grad_focal: float");

template <typename CPUDevice>
void ComputeDepthGradKernel<CPUDevice>::operator() (const CPUDevice &d,
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

        const int input_col = index % input_width;
        const int input_row = ((index - input_col) / input_width) % input_height;
        const int batch = ((index - input_row * input_width - input_col) / (input_width*input_height)) % batch_size;

        unsigned int input_index =
            batch*input_height*input_width*4 + input_row*input_width*4 + input_col*4;

        grad_input[input_index + 0] = 0.0f;
        grad_input[input_index + 1] = 0.0f;
        grad_input[input_index + 2] = 0.0f;
        grad_input[input_index + 3] = 0.0f;

        float fo = focal[batch];

        for(unsigned int r = 0; r < num_threads_row; ++r)
        {
            for(unsigned int cc = 0; cc < num_threads_col; ++cc)
            {
                unsigned int col = input_col * num_threads_col + cc;
                unsigned int row = input_row * num_threads_row + r;

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

template <typename Device>
class ComputeDepthGradOp : public OpKernel {
    public:
        explicit ComputeDepthGradOp(OpKernelConstruction* context) : OpKernel(context) {
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

            const int batch_size = input_shape.dim_size(0);
            const int input_height = input_shape.dim_size(1);
            const int input_width = input_shape.dim_size(2);

            ComputeDepthGradKernel<Device>()(context->eigen_device<Device>(),
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
REGISTER_KERNEL_BUILDER(Name("ComputeDepthGrad").Device(DEVICE_CPU), ComputeDepthGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
namespace functor {
template <>                                                          
void ComputeDepthGradKernel<GPUDevice>::operator()(const GPUDevice& d,
                                                   const int batch_size, 
                                                   const int input_height, 
                                                   const int input_width,
                                                   const int depth_height,
                                                   const int depth_width,
                                                   const float* depth_grad,
                                                   const float* input, 
                                                   const float* focal, 
                                                   float* depth);
extern template struct ComputeDepthGradKernel<GPUDevice>;
}
template<>
class ComputeDepthGradOp<GPUDevice> : public OpKernel {
    public:
        explicit ComputeDepthGradOp(OpKernelConstruction* context) : OpKernel(context) {
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

            functor::ComputeDepthGradKernel<GPUDevice>()(context->eigen_device<GPUDevice>(),
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
REGISTER_KERNEL_BUILDER(Name("ComputeDepthGrad").Device(DEVICE_GPU), ComputeDepthGradOp<GPUDevice>);
#endif
