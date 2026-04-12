#include <cassert>
#include "quantization.cuh"

#define MASK_ALL 0xffffffff
#define MIN_SCALE 1e-5

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ROUND(f) static_cast<int8_t>((f) >= 0 ? (f) + 0.5f : (f) - 0.5f)

template <typename T>
struct Matrix {
    T* data;
    size_t rows;
    size_t cols;
    size_t stride;

    __device__ inline const T& operator()(size_t y, size_t x) const {
        assert(x < cols && y < rows);
        return data[y * stride + x];  // allways row major
    }

    __device__ inline T& operator()(size_t y, size_t x) {
        assert(x < cols && y < rows);
        return data[y * stride + x];  // allways row major
    }
};

void __global__ QuantizationKernel(Matrix<const float> d_input, Matrix<int8_t> d_output,
                                   const float* d_balance_factors, float* d_out_scales) {
    // compute out_scales one element per warp using strided loop
    // __syncwarp();
    // calculate per thread output elements using strided loop

    // 32 threads / warp
    // 16 warps / block? so we have 512 threads total

    int tx = threadIdx.x % 32;
    int ty = threadIdx.x / 32;

    int by = blockIdx.x;

    int y = by * 16 + ty;

    // max_x | Wx'y' + Sx'|
    float max_abs = 0.0f;
    const int offset = 32;
    for (int x = tx; x < d_input.cols; x += offset) {
        if (y < d_input.rows) {
            float new_val = fabsf(d_input(y, x) + d_balance_factors[x]);
            max_abs = MAX(max_abs, new_val);
        }
    }

    // reduce inside warp to get each out_scale factors
    __syncwarp();

    for (size_t shift = 16; shift > 0; shift /= 2) {
        auto new_max_abs = __shfl_down_sync(MASK_ALL, max_abs, shift);
        max_abs = MAX(max_abs, new_max_abs);
    }

    if (tx == 0) {
        if (y < d_input.rows) {
            max_abs = 127.0 / MAX(max_abs, MIN_SCALE);
            d_out_scales[y] = max_abs;
        }
    }

    // only need to sync warps, because thread only needs stuff inside its warp
    // __syncwarp();

    // broadcast scaler to each thread in warp
    max_abs = __shfl_sync(MASK_ALL, max_abs, 0);
    // quanisize each element
    for (int x = tx; x < d_input.cols; x += offset) {
        if (y < d_input.rows) {
            d_output(y, x) = ROUND((d_input(y, x) + d_balance_factors[x]) * max_abs);
        }
    }
}

void Quantization(size_t rows, size_t cols, const float* d_input_matrix,
                  const float* d_balance_factors, size_t input_stride, size_t out_stride,
                  int8_t* d_out, float* d_out_scales) {
    // YOUR CODE HERE
    // NB: no need to do any allocations here
    // NB: no explicit cudaDeviceSynchronize is required here
    Matrix<const float> input{d_input_matrix, rows, cols, input_stride};
    Matrix<int8_t> output{d_out, rows, cols, out_stride};

    QuantizationKernel<<<(rows + 16 - 1) / 16, 32 * 16>>>(input, output, d_balance_factors,
                                                          d_out_scales);
}
