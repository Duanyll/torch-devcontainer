#include "kernel.cuh"
#include "utils.h"

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) { return a + b; }

torch::Tensor my_add_cuda(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda(), "Tensors must be on a CUDA device");
  TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same device");
  TORCH_CHECK(a.dtype() == b.dtype() && a.dtype() == torch::kFloat32,
              "Tensors must be of type float32");
  TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "Tensors must be contiguous");

  torch::Tensor result = torch::empty_like(a);
  my_add_impl(a.data_ptr<float>(), b.data_ptr<float>(),
              result.data_ptr<float>(), a.numel());
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_add", &my_add, "A function that adds two tensors");
  m.def("my_add_cuda", &my_add_cuda,
        "A function that adds two tensors on CUDA with custom kernel");
}