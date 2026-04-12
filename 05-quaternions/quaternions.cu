#include "quaternions.cuh"

#include <cassert>

#include <cuda_runtime.h>

template <int Cols>
__global__ void QuanternionsReduceKernel(size_t rows, const Quaternion* __restrict__ inp,
                                         size_t inp_stride, Quaternion* __restrict__ out) {
    int tx = threadIdx.x % 32;
    int ty = threadIdx.x / 32;

    int by = blockIdx.x;

    int y = by * 16 + ty;

    if (y < rows) {
        Quaternion row_total = {1.0f, 0.0f, 0.0f, 0.0f};
        constexpr int kTiles = Cols / 32;
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
            Quaternion q1 = inp[y * inp_stride + tile * 32 + tx];

#pragma unroll
            for (int shift = 1; shift < 32; shift *= 2) {
                Quaternion q2;
                q2.a = __shfl_down_sync(0xffffffff, q1.a, shift);
                q2.b = __shfl_down_sync(0xffffffff, q1.b, shift);
                q2.c = __shfl_down_sync(0xffffffff, q1.c, shift);
                q2.d = __shfl_down_sync(0xffffffff, q1.d, shift);

                if (tx + shift < 32) {
                    auto tmp = QuaternionMultiplier{}(q1, q2);
                    q1.a = tmp.a;
                    q1.b = tmp.b;
                    q1.c = tmp.c;
                    q1.d = tmp.d;
                }
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
    dim3 block(32 * 16);
    dim3 grid((rows + 15) / 16);
    switch (cols) {
        case 1024:
            QuanternionsReduceKernel<1024><<<grid, block, 0, stream>>>(rows, inp, inp_stride, out);
            break;
        case 2048:
            QuanternionsReduceKernel<2048><<<grid, block, 0, stream>>>(rows, inp, inp_stride, out);
            break;
        case 4096:
            QuanternionsReduceKernel<4096><<<grid, block, 0, stream>>>(rows, inp, inp_stride, out);
            break;
        default:
            assert(false);
    }
}
