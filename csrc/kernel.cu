#include "kernel.cuh"

__global__ void my_add_kernel(float* a, float* b, float* res, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    res[idx] = a[idx] + b[idx];
  }
}

void my_add_impl(float* a, float* b, float* res, int64_t n) {
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  my_add_kernel<<<numBlocks, blockSize>>>(a, b, res, n);
}