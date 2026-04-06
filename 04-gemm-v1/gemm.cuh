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

#define TILE_DIM 64
#define TILE_INNER 8
#define TM 8

__global__ void GEMMKernel(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c,
                           DeviceMatrix d, float alpha, float beta) {
    // A: [M, N]
    // B: [N, K]
    // C: [M, K]

    __shared__ __half a_tile[TILE_DIM][TILE_INNER];
    __shared__ __half b_tile[TILE_INNER][TILE_DIM];

    // one block calculates one tile in C, starting at
    const size_t tile_x = blockIdx.x * TILE_DIM;
    const size_t tile_y = blockIdx.y * TILE_DIM;

    // 1D thread indexing
    const size_t thread_x = threadIdx.x % TILE_DIM;  // column in output tile
    const size_t thread_y = threadIdx.x / TILE_DIM;  // row-group in output tile

    const size_t inner_ax = threadIdx.x % TILE_INNER;
    const size_t inner_ay = threadIdx.x / TILE_INNER;
    const size_t inner_bx = threadIdx.x % TILE_DIM;
    const size_t inner_by = threadIdx.x / TILE_DIM;

    // accumulate TM values instead of 1 to remove smem bottleneck
    float tmp[TM] = {0.0f};

    // stride by TILE_INNER instead of TILE_DIM
    for (size_t i = 0; i < a.cols; i += TILE_INNER) {
        // coalesced fetch A, B matrixes to tiles
        {
            size_t ay = tile_y + inner_ay;
            size_t ax = i + inner_ax;
            __half val = (ay < a.rows && ax < a.cols) ? a.At(ay, ax) : __float2half(0.0f);
            a_tile[inner_ay][inner_ax] = val;
        }
        {
            size_t by = i + inner_by;
            size_t bx = tile_x + inner_bx;
            __half val = (by < b.rows && bx < b.cols) ? b.At(by, bx) : __float2half(0.0f);
            b_tile[inner_by][inner_bx] = val;
        }

        __syncthreads();

        // dot products inside tiles
        // res[y, x] = <A[y, :], B[:, x]>
        // 1D tiling
        for (size_t j = 0; j < TILE_INNER; ++j) {
            __half tmp_b = b_tile[j][thread_x];
            for (size_t k = 0; k < TM; ++k) {
                tmp[k] += static_cast<float>(a_tile[thread_y * TM + k][j] * tmp_b);
            }
        }

        // sync threds to prevent fast threads to load new data to smem
        // while slow threads still computing dot product
        __syncthreads();
    }

    // write TM results per thread
    for (size_t k = 0; k < TM; ++k) {
        size_t y = tile_y + thread_y * TM + k;
        size_t x = tile_x + thread_x;
        // D[y, x] = a * <A[y, :], B[:, x]> + b * C[y, x]
        if (y < d.rows && x < d.cols) {
            float t1 = alpha * tmp[k];
            float t2 = beta * static_cast<float>(c.At(y, x));

            d.At(y, x) = static_cast<__half>(t1 + t2);
        }
    }
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(c.cols == d.cols && c.rows == d.rows);

    // 1D block: TILE_DIM/TM * TILE_DIM = 8 * 64 = 512 threads
    dim3 block(TILE_DIM * TILE_INNER);  // = (TILE_DIM / TM) * TILE_DIM = 512

    dim3 grid((d.cols + TILE_DIM - 1) / TILE_DIM, (d.rows + TILE_DIM - 1) / TILE_DIM);

    GEMMKernel<<<grid, block>>>(a, b, c, d, alpha, beta);
    CheckStatus(cudaGetLastError());
    CheckStatus(cudaDeviceSynchronize());
}

// tile row can be processed with one warp easy peasy in one clock cycle
// #define TILE_DIM 32

__global__ void GEMMKernelOld(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c,
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

void GEMMOld(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
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
