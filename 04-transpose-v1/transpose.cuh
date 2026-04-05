#include <cuda_fp16.h>
#include <cassert>

#include <cuda_helpers.h>

// tile row can be processed with one warp easy peasy in one clock cycle
#define TILE_DIM 32

// amount of warps = TILE_DIM / BLOCK_ROWS
// how to determine optimal amount of warps inside one block??
#define BLOCK_ROWS 4

template <typename T>
struct Matrix {
    T* data;
    size_t stride;
    size_t num_rows;
    size_t num_cols;

    __device__ T& operator()(size_t y, size_t x) {
        // assert(x < num_cols && y < num_rows);
        return data[y * stride + x];
    }
};

// to escape bank conflicts when loading col from smem to output row
#define PAD 1
__global__ void TransposeKernel(Matrix<__half> input, Matrix<__half> output) {
    __shared__ __half block_data[TILE_DIM * (TILE_DIM + PAD)];
    Matrix<__half> block{block_data, (TILE_DIM + PAD), TILE_DIM, (TILE_DIM + PAD)};

    size_t x_block = static_cast<size_t>(blockIdx.x) * TILE_DIM;
    size_t y_block = static_cast<size_t>(blockIdx.y) * TILE_DIM;

    size_t x;
    size_t y;

    // load rows from input to tile
    // one warp should load full row, so each thread locks one col of the tile
    x = x_block + threadIdx.x;
    y = y_block + threadIdx.y;
    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < input.num_cols && y + j < input.num_rows) {
            block(threadIdx.y + j, threadIdx.x) = input(y + j, x);
        }
    }

    __syncthreads();

    // load col from tile to row in output
    x = y_block + threadIdx.x;
    y = x_block + threadIdx.y;
    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < input.num_cols && x < input.num_rows) {
            output(y + j, x) = block(threadIdx.x, threadIdx.y + j);
        }
    }
}

void TransposeDevice(const __half* input_device, size_t input_stride, __half* output_device,
                     size_t output_stride, size_t num_rows, size_t num_cols) {
    // assert(num_cols % TILE_DIM == 0 && num_rows % TILE_DIM == 0);
    static_assert(TILE_DIM % BLOCK_ROWS == 0);
    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid((num_cols + TILE_DIM - 1) / TILE_DIM, (num_rows + TILE_DIM - 1) / TILE_DIM);

    Matrix<__half> input{const_cast<__half*>(input_device), input_stride, num_rows, num_cols};
    Matrix<__half> output{output_device, output_stride, num_cols, num_rows};

    TransposeKernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}
