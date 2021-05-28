#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> voronoi_interp_cuda_forward(
    torch::Tensor coords, // const torch::Tensor& ?
    torch::Tensor values,
    int H,
    int W);

torch::Tensor voronoi_interp_cuda_backward(
    torch::Tensor grad_output_image,
    torch::Tensor nearest_idxs,
    int N);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> voronoi_interp_forward(
    torch::Tensor coords,
    torch::Tensor values,
    int H,
    int W) {

  CHECK_INPUT(coords);
  CHECK_INPUT(values);

  return voronoi_interp_cuda_forward(coords, values, H, W);
}

torch::Tensor voronoi_interp_backward(torch::Tensor grad_output_image, torch::Tensor nearest_idxs, int N) {
  CHECK_INPUT(grad_output_image);
  CHECK_INPUT(nearest_idxs);

  return voronoi_interp_cuda_backward(grad_output_image, nearest_idxs, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voronoi_interp_forward", &voronoi_interp_forward, "voronoi_interp forward (CUDA)");
  m.def("voronoi_interp_backward", &voronoi_interp_backward, "voronoi_interp backward (CUDA)");
}
