#include "quaternions.cuh"

#include <cassert>

#include <cuda_runtime.h>

#define MASK_ALL 0xffffffff

__global__ void QuanternionsReduceKernel(size_t rows, size_t cols, const Quaternion* inp,
                                         size_t inp_stride, Quaternion* out) {
    int tx = threadIdx.x % 32;
    int ty = threadIdx.x / 32;

    int by = blockIdx.x;

    int y = by * 16 + ty;

    // prod in row
    if (y < rows) {
        Quaternion row_total = {1.0f, 0.0f, 0.0f, 0.0f};
        for (int x = tx; x < cols; x += 32) {
            Quaternion q1 = inp[y * inp_stride + x];

            // reduce the 32 elements within this warp tile
#pragma unroll
            for (int shift = 1; shift < 32; shift *= 2) {

                Quaternion q2;
                q2.a = __shfl_down_sync(0xffffffff, q1.a, shift);
                q2.b = __shfl_down_sync(0xffffffff, q1.b, shift);
                q2.c = __shfl_down_sync(0xffffffff, q1.c, shift);
                q2.d = __shfl_down_sync(0xffffffff, q1.d, shift);

                auto tmp = QuaternionMultiplier{}(q1, q2);
                q1.a = tmp.a;
                q1.b = tmp.b;
                q1.c = tmp.c;
                q1.d = tmp.d;
            }

            if (tx == 0) {
                row_total = QuaternionMultiplier{}(row_total, q1);
            }
        }

        if (tx == 0) {
            out[y] = row_total;
        }
    }
}

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {
    // YOUR CODE HERE
    // NB: no need to do any allocations here
    // NB: no explicit cudaDeviceSynchronize is required here
    dim3 block(32 * 16);
    dim3 grid((rows + 15) / 16);
    QuanternionsReduceKernel<<<grid, block, 0, stream>>>(rows, cols, inp, inp_stride, out);
}
