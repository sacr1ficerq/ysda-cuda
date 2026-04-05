#pragma once

#include <cassert>
#include <cuda_fp16.h>

#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Distance in elements between first values of consecutive rows/columns
    MatrixLayout layout;

    __host__ __device__ inline __half& At(size_t y, size_t x) const {
        const size_t idx =
            this->layout == MatrixLayout::RowMajor ? y * this->stride + x : y + x * this->stride;
        return this->data[idx];
    }
};

__global__ void GEMMKernel(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c,
                           DeviceMatrix d, float alpha, float beta) {
    // A: [M, N]
    // B: [N, K]
    // C: [M, K]
    const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= c.cols || y >= c.rows) {
        return;
    }

    // D[y, x] = <A[y, :], B[:, x]> + C[y, x]
    assert(a.cols == b.rows);
    size_t n = a.cols;
    float s = 0;
    for (size_t i = 0; i < n; ++i) {
        s += static_cast<float>(a.At(y, i) * b.At(i, x));
    }

    float t1 = alpha * s;
    float t2 = beta * static_cast<float>(c.At(y, x));

    d.At(y, x) = static_cast<__half>(t1 + t2);
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(c.cols == d.cols && c.rows == d.rows);

    size_t total_width = c.cols;
    size_t total_height = c.rows;

    const size_t block_width = 32;
    const size_t block_height = 32;

    dim3 block(block_width, block_height, 1);

    size_t grid_width = (total_width + block_width - 1) / block_width;
    size_t grid_height = (total_height + block_height - 1) / block_height;

    dim3 grid(grid_width, grid_height, 1);

    GEMMKernel<<<grid, block>>>(a, b, c, d, alpha, beta);
    CheckStatus(cudaGetLastError());
    CheckStatus(cudaDeviceSynchronize());
}
