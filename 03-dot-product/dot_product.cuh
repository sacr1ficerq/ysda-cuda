#pragma once

#include <cassert>
#include <cuda_helpers.h>

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    size_t thread_size = 512;
    size_t block_size = (num_elements + thread_size - 1) / thread_size;
    return block_size * sizeof(double);
}

__global__ void BlockDotProductKernel(const float* lhs_device, const float* rhs_device,
                                      size_t num_elements, double* workspace_device) {
    const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

    __shared__ double shared_buffer[512];

    if (threadIdx.x == 0) {
        workspace_device[blockIdx.x] = 0;
    }

    if (index < num_elements) {
        shared_buffer[threadIdx.x] = static_cast<double>(lhs_device[index]) * rhs_device[index];
    }
    __syncthreads();

    // reduce inside each kernel and put inside workspace[blockIdx.x]
    if (index < num_elements) {
        atomicAdd(workspace_device + blockIdx.x, shared_buffer[threadIdx.x]);
    }
}

__global__ void SumKernel(size_t num_elements, double* workspace_device, float* out_device) {
    const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index < num_elements) {
        atomicAdd(out_device, workspace_device[index]);
    }
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    size_t thread_size = 512;
    size_t block_size = (num_elements + thread_size - 1) / thread_size;
    double* workspace_device_double = reinterpret_cast<double*>(workspace_device);

    BlockDotProductKernel<<<block_size, thread_size>>>(lhs_device, rhs_device, num_elements,
                                                       workspace_device_double);
    cudaDeviceSynchronize();

    num_elements = block_size;
    thread_size = 512;
    block_size = (num_elements + thread_size - 1) / thread_size;

    SumKernel<<<block_size, thread_size>>>(num_elements, workspace_device_double, out_device);
    cudaDeviceSynchronize();
}
