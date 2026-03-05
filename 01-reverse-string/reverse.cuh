#pragma once

#include <cstddef>
#include <cuda_helpers.h>
#include <cuda_runtime.h>
#include <iostream>

#define N_BLOCKS(N, M) ((N) + (M) - 1) / (M)

__global__ void ReverseKernel(char* s, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n / 2) {
        return;
    }
    // thrust::swap(s[idx], s[n - idx - 1]);
    char temp = s[idx];
    s[idx] = s[n - 1 - idx];
    s[n - 1 - idx] = temp;
}

void ReverseDeviceStringInplace(char* str, size_t length) {
    if (length == 0 || length == 1) {
        return;
    }
    size_t n_threads = 256;
    size_t n_blocks = N_BLOCKS(length / 2, n_threads);
    ReverseKernel<<<n_blocks, n_threads>>>(str, length);
    cudaDeviceSynchronize();
}
