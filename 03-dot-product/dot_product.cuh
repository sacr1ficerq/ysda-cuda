#pragma once

#include <cassert>
#include <cuda_helpers.h>

#define THREAD_SIZE_DEG 8

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    return 0;
    size_t thread_size = 512;
    size_t block_size = (num_elements + thread_size - 1) / thread_size;
    return block_size * sizeof(float);
}

__global__ void BlockDotProductKernel(const float* lhs_device, const float* rhs_device,
                                      size_t num_elements, float* out_device) {
    const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    __shared__ float shared_buffer[1 << THREAD_SIZE_DEG];

    if (index >= num_elements) {
        shared_buffer[threadIdx.x] = 0;
        return;
    }

    shared_buffer[threadIdx.x] = lhs_device[index] * rhs_device[index];
    __syncthreads();

    // half the sum
    size_t cnt = 1 << (THREAD_SIZE_DEG - 1);  // 512 / 2
    while (cnt != 0) {
        if (threadIdx.x < cnt) {
            shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x + cnt];
        }
        cnt >>= 1;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_device, shared_buffer[0]);
    }
}

__global__ void SumKernel(size_t num_elements, float* workspace_device, float* out_device) {
    __shared__ float shared_buffer[1024];

    if (threadIdx.x >= num_elements) {
        shared_buffer[threadIdx.x] = 0;
        return;
    }
    shared_buffer[threadIdx.x] = workspace_device[threadIdx.x];
    __syncthreads();

    size_t cnt = 1 << 9;  // 1024 / 2
    while (cnt != 0) {
        if (threadIdx.x < cnt) {
            shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x + cnt];
        }
        cnt >>= 1;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_device = shared_buffer[0];
    }
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    size_t thread_size = 1 << THREAD_SIZE_DEG;
    size_t block_size = (num_elements + thread_size - 1) / thread_size;
    (void)workspace_device;

    BlockDotProductKernel<<<block_size, thread_size>>>(lhs_device, rhs_device, num_elements,
                                                       out_device);
}
