#pragma once

#include <cassert>
#include <cuda_helpers.h>

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    size_t thread_size = 512;
    size_t block_size = (num_elements + thread_size - 1) / thread_size;
    return block_size * sizeof(float);
}

__global__ void BlockDotProductKernel(const float* lhs_device, const float* rhs_device,
                                      size_t num_elements, float* workspace_device) {
    const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    __shared__ float shared_buffer[512];

    if (index > num_elements) {
        shared_buffer[threadIdx.x] = 0;
        return;
    }

    if (threadIdx.x == 0) {
        workspace_device[blockIdx.x] = 0;
    }

    shared_buffer[threadIdx.x] = lhs_device[index] * rhs_device[index];
    __syncthreads();

    // half the sum
    size_t cnt = 1 << 8;  // 512 / 2
    while (cnt != 0) {
        if (threadIdx.x < cnt) {
            shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x + cnt];
        }
        cnt >>= 1;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        workspace_device[blockIdx.x] = shared_buffer[0];
    }
}

__global__ void SumKernel(size_t num_elements, float* workspace_device, float* out_device) {
    const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index < num_elements) {
        atomicAdd(out_device, workspace_device[index]);
    }
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    size_t thread_size = 512;
    size_t block_size = (num_elements + thread_size - 1) / thread_size;

    BlockDotProductKernel<<<block_size, thread_size>>>(lhs_device, rhs_device, num_elements,
                                                       workspace_device);
    cudaDeviceSynchronize();

    num_elements = block_size;
    thread_size = 512;
    block_size = (num_elements + thread_size - 1) / thread_size;

    SumKernel<<<block_size, thread_size>>>(num_elements, workspace_device, out_device);
    cudaDeviceSynchronize();
}

