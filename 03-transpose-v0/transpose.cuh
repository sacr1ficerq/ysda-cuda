#include <cuda_fp16.h>

#include <cuda_helpers.h>

__global__ void naiveTransposeKernel(const __half* input, size_t input_stride, __half* output,
                                     size_t output_stride, size_t num_rows, size_t num_cols) {
    size_t x = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockDim.y) * blockIdx.y + threadIdx.y;

    if (y <= num_rows - 1 && x <= num_cols - 1) {
        size_t input_idx = y * input_stride + x;
        size_t output_idx = x * output_stride + y;
        output[output_idx] = input[input_idx];
    }
}

void TransposeDevice(const __half* input_device, size_t input_stride, __half* output_device,
                     size_t output_stride, size_t num_rows, size_t num_cols) {
    dim3 block(32, 32, 1);
    size_t grid_width = (num_cols + 32 - 1) / 32;
    size_t grid_height = (num_rows + 32 - 1) / 32;
    dim3 grid(grid_width, grid_height);

    naiveTransposeKernel<<<grid, block>>>(input_device, input_stride, output_device, output_stride,
                                          num_rows, num_cols);
    cudaDeviceSynchronize();
}
