#pragma once

#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#define _DEBUG 1
#else
#include <torch/extension.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

// Print a matrix on the device stored in row-major order
template <typename T>
void print_device_matrix(const char *name, T *d_A, int m, int n) {
  T *h_A = new T[m * n];
  cudaMemcpy(h_A, d_A, m * n * sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << name << " = " << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < m; i++) {
    std::cout << "  ";
    for (int j = 0; j < n; j++) {
      std::cout << std::setw(8) << h_A[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
  delete[] h_A;
}

template <typename T>
void print_device_vector(const char *name, T *d_A, int n) {
  T *h_A = new T[n];
  cudaMemcpy(h_A, d_A, n * sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << name << " = " << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < n; i++) {
    std::cout << "  ";
    std::cout << std::setw(8) << h_A[i] << " ";
    std::cout << std::endl;
  }
  delete[] h_A;
}

template <typename T>
void print_host_matrix(const char *name, T *h_A, int m, int n) {
  std::cout << name << " = " << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < m; i++) {
    std::cout << "  ";
    for (int j = 0; j < n; j++) {
      std::cout << std::setw(8) << h_A[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T> void print_host_vector(const char *name, T *h_A, int n) {
  std::cout << name << " = " << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < n; i++) {
    std::cout << "  ";
    std::cout << std::setw(8) << h_A[i] << " ";
    std::cout << std::endl;
  }
}

#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))