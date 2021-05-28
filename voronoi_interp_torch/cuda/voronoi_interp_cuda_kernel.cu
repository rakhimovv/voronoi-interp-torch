#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_distance(scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2) {
    const auto x_diff = (x1 - x2);
    const auto y_diff = (y1 - y2);
    const auto dist2 = x_diff * x_diff + y_diff * y_diff;
    return dist2;
}


template <typename scalar_t>
__global__ void voronoi_interp_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_image,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> nearest_idxs) {

    const int pixel_pos_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_pos_y = blockDim.y * blockIdx.y + threadIdx.y;

    int64_t nearest_idx = -1;
    scalar_t h = static_cast<scalar_t>(output_image.size(1));
    scalar_t w = static_cast<scalar_t>(output_image.size(2));
    scalar_t min_dist2 = h * h + w * w + 1.0;

    if (pixel_pos_x >= 0 && pixel_pos_y >= 0 && pixel_pos_x < output_image.size(2) && pixel_pos_y < output_image.size(1)) {
        for (int i = 0; i < coords.size(0); i++) {
            const scalar_t dist2 = compute_distance(
                static_cast<scalar_t>(pixel_pos_x),
                static_cast<scalar_t>(pixel_pos_y),
                static_cast<scalar_t>(coords[i][0]),
                static_cast<scalar_t>(coords[i][1]));
            if (dist2 < min_dist2) {
                min_dist2 = dist2;
                nearest_idx = i;
            }
        }
        nearest_idxs[pixel_pos_y][pixel_pos_x] = nearest_idx;
        if (nearest_idx != -1) {
            for (int c = 0; c < output_image.size(0); c++) {
                output_image[c][pixel_pos_y][pixel_pos_x] = values[nearest_idx][c];
            }
        }
    }
}

template <typename scalar_t>
__global__ void voronoi_interp_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output_image,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> nearest_idxs,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_values) {

    const int pixel_pos_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_pos_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (pixel_pos_x >= 0 && pixel_pos_y >= 0 && pixel_pos_x < grad_output_image.size(2) && pixel_pos_y < grad_output_image.size(1)) {
        for (int c = 0; c < grad_output_image.size(0); c++) {
            atomicAdd(&grad_values[nearest_idxs[pixel_pos_y][pixel_pos_x]][c], grad_output_image[c][pixel_pos_y][pixel_pos_x]);
        }
    }
}
} // namespace

std::vector<torch::Tensor> voronoi_interp_cuda_forward(
    torch::Tensor coords,
    torch::Tensor values,
    int H,
    int W) {

    auto output_image = torch::zeros({3, H, W}, values.options()).contiguous();
    auto nearest_idxs = torch::zeros({H, W}, torch::dtype(torch::kLong).device(values.device()).requires_grad(false)).contiguous();

    {
        const dim3 threads(16, 16);
        const dim3 blocks((output_image.size(2) + 16 - 1) / 16, (output_image.size(1) + 16 - 1) / 16);

        AT_DISPATCH_FLOATING_TYPES(values.type(), "voronoi_interp_cuda_forward_kernel", ([&] {
        voronoi_interp_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            coords.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            nearest_idxs.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    return {output_image, nearest_idxs};
}

torch::Tensor voronoi_interp_cuda_backward(
    torch::Tensor grad_output_image,
    torch::Tensor nearest_idxs,
    int N) {

    auto grad_values = torch::zeros({N, grad_output_image.size(0)}, grad_output_image.options()).contiguous();

    {
        const dim3 threads(16, 16);
        const dim3 blocks((grad_output_image.size(2) + 16 - 1) / 16, (grad_output_image.size(1) + 16 - 1) / 16);

        AT_DISPATCH_FLOATING_TYPES(grad_values.type(), "voronoi_interp_cuda_backward_kernel", ([&] {
        voronoi_interp_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            nearest_idxs.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    return grad_values;
}
