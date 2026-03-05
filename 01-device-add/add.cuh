#pragma once

#include <vector>
#include <cassert>
#include <cuda_helpers.h>

#define C(X) (void)(X)

float* AllocDeviceVector(size_t num_elements) {
    float* src_device = nullptr;
    cudaMalloc(&src_device, num_elements * sizeof(float));
    return src_device;
}

void FreeDeviceVector(float* device_ptr) {
    cudaFree(device_ptr);
}

void CopyHostVectorToDevice(const std::vector<float>& vector_host, float* dst_device_ptr) {
    auto copy_kind = cudaMemcpyHostToDevice;
    cudaMemcpy(dst_device_ptr, vector_host.data(), vector_host.size() * sizeof(float), copy_kind);
}

std::vector<float> CopyDeviceVectorToHost(const float* ptr_device, size_t num_elements) {
    auto copy_kind = cudaMemcpyDeviceToHost;
    std::vector<float> dst_vector(num_elements);
    cudaMemcpy(dst_vector.data(), ptr_device, num_elements * sizeof(float), copy_kind);
    return dst_vector;
}

__global__ void AddKernel(const float* left, const float* right, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // we are not out of bounds
        out[idx] = left[idx] + right[idx];
    }
}

constexpr size_t GetBlocks(const size_t n, const size_t block_size) {
    return (n + block_size - 1) / block_size;
}

void AddDeviceVectors(const float* left_device, const float* right_device, float* out_device,
                      size_t num_elements) {
    size_t threads = 256;
    size_t blocks = GetBlocks(num_elements, threads);
    assert(blocks * threads >= num_elements);
    AddKernel<<<blocks, threads>>>(left_device, right_device, out_device, num_elements);
}
