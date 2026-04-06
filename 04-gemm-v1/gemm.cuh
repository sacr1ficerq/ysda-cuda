#pragma once

#include <cassert>
#include <cuda_fp16.h>

#include <cuda_helpers.h>

// tile row can be processed with one warp easy peasy in one clock cycle
#define TILE_DIM 32

// amount of warps = TILE_DIM / BLOCK_ROWS
// how to determine optimal amount of warps inside one block??
// #define BLOCK_ROWS 4

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

    __shared__ __half a_tile[TILE_DIM][TILE_DIM];
    __shared__ __half b_tile[TILE_DIM][TILE_DIM + 1];
    // one block calculates one tile in C, starting at
    const size_t tile_x = blockIdx.x * TILE_DIM;
    const size_t tile_y = blockIdx.y * TILE_DIM;

    float tmp = 0.0;
    // we move tile on x axis in matrix A and y axis on matrix B
    for (size_t i = 0; i < a.cols; i += TILE_DIM) {
        // coords of cell for each thread inside block
        size_t ax = i + threadIdx.x;
        size_t ay = tile_y + threadIdx.y;

        size_t bx = tile_x + threadIdx.x;
        size_t by = i + threadIdx.y;

        // coalesced fetch A, B matrixes to tiles
        if (ay < a.rows && ax < a.cols) {
            a_tile[threadIdx.y][threadIdx.x] = a.At(ay, ax);
        } else {
            a_tile[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        if (by < b.rows && bx < b.cols) {
            b_tile[threadIdx.y][threadIdx.x] = b.At(by, bx);
        } else {
            b_tile[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        // wait for all threads to fill smem
        __syncthreads();

        // dot products inside tiles
        // res[y, x] = <A[y, :], B[:, x]>
        for (size_t j = 0; j < TILE_DIM; ++j) {
            tmp += static_cast<float>(a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x]);
        }

        // sync threds to prevent fast threads to load new data to smem
        // while slow threads still computing dot product
        __syncthreads();
    }

    // D[y, x] = <A[y, :], B[:, x]> + C[y, x]
    size_t x = tile_x + threadIdx.x;
    size_t y = tile_y + threadIdx.y;
    if (y < d.rows && x < d.cols) {
        float t1 = alpha * tmp;
        float t2 = beta * static_cast<float>(c.At(y, x));

        d.At(y, x) = static_cast<__half>(t1 + t2);
    }
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(c.cols == d.cols && c.rows == d.rows);

    size_t total_width = c.cols;
    size_t total_height = c.rows;

    dim3 block(TILE_DIM, TILE_DIM, 1);

    size_t grid_width = (total_width + TILE_DIM - 1) / TILE_DIM;
    size_t grid_height = (total_height + TILE_DIM - 1) / TILE_DIM;

    dim3 grid(grid_width, grid_height, 1);

    GEMMKernel<<<grid, block>>>(a, b, c, d, alpha, beta);
    CheckStatus(cudaGetLastError());
    CheckStatus(cudaDeviceSynchronize());
}
