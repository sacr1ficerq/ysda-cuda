#include "quaternions.cuh"

#include <cassert>

#include <cuda_runtime.h>

template <int Cols>
__global__ void QuanternionsReduceKernel(size_t rows, const Quaternion* __restrict__ inp,
                                         size_t inp_stride, Quaternion* __restrict__ out) {
    int tx = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    constexpr int num_warps = 256 / 32;

    int y = blockIdx.x;

    if (y < rows) {
        Quaternion warp_total = {1.0f, 0.0f, 0.0f, 0.0f};
        constexpr int kTiles = Cols / 32;
        constexpr int tiles_per_warp = kTiles / num_warps;

        int start_tile = warp_id * tiles_per_warp;
        int end_tile = start_tile + tiles_per_warp;

#pragma unroll 4
        for (int tile = start_tile; tile < end_tile; ++tile) {
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
                warp_total = QuaternionMultiplier{}(warp_total, q1);
            }
        }

        __shared__ Quaternion smem[num_warps];

        if (tx == 0) {
            smem[warp_id] = warp_total;
        }

        __syncthreads();

        if (warp_id == 0) {
            Quaternion q_final = {1.0f, 0.0f, 0.0f, 0.0f};
            if (tx < num_warps) {
                q_final = smem[tx];
            }

#pragma unroll
            for (int shift = 1; shift < 32; shift *= 2) {
                Quaternion q2;
                q2.a = __shfl_down_sync(0xffffffff, q_final.a, shift);
                q2.b = __shfl_down_sync(0xffffffff, q_final.b, shift);
                q2.c = __shfl_down_sync(0xffffffff, q_final.c, shift);
                q2.d = __shfl_down_sync(0xffffffff, q_final.d, shift);

                if (tx + shift < 32) {
                    auto tmp = QuaternionMultiplier{}(q_final, q2);
                    q_final.a = tmp.a;
                    q_final.b = tmp.b;
                    q_final.c = tmp.c;
                    q_final.d = tmp.d;
                }
            }

            if (tx == 0) {
                out[y] = q_final;
            }
        }
    }
}

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {
    if (rows == 0) {
        return;
    }

    dim3 block(256);
    dim3 grid(rows);
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
