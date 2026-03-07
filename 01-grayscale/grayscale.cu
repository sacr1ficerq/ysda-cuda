#include "grayscale.cuh"

Image AllocHostImage(size_t width, size_t height, size_t channels) {
    size_t image_size = width * height * channels;
    auto image_data = static_cast<uint8_t*>(malloc(image_size * sizeof(uint8_t)));
    Image image{.pixels = image_data,
                .width = width,
                .height = height,
                .stride = width * channels * sizeof(uint8_t),
                .channels = channels};
    return image;
}

Image AllocDeviceImage(size_t width, size_t height, size_t channels) {
    size_t data_width = width * channels;

    uint8_t* image_data = nullptr;
    size_t pitch_bytes;

    cudaMallocPitch(&image_data, &pitch_bytes, data_width * sizeof(uint8_t), height);

    Image image{.pixels = image_data,
                .width = width,
                .height = height,
                .stride = pitch_bytes,
                .channels = channels};
    return image;
}

void CopyImageHostToDevice(const Image& src_host, Image& dst_device) {
    // assuming dst_device has pitch
    auto copy_kind = cudaMemcpyHostToDevice;
    cudaMemcpy2D(dst_device.pixels, dst_device.stride, src_host.pixels, src_host.stride,
                 src_host.width * src_host.channels, src_host.height, copy_kind);
    dst_device.height = src_host.height;
    dst_device.width = src_host.width;
    dst_device.channels = src_host.channels;
}

void CopyImageDeviceToHost(const Image& src_device, Image& dst_host) {
    auto copy_kind = cudaMemcpyDeviceToHost;
    cudaMemcpy2D(dst_host.pixels, dst_host.stride, src_device.pixels, src_device.stride,
                 src_device.width * src_device.channels, src_device.height, copy_kind);
    dst_host.height = src_device.height;
    dst_host.width = src_device.width;
    dst_host.channels = src_device.channels;
}

__global__ void GrayscaleKernel(const Image rgb, Image gray) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= static_cast<int>(rgb.width) || y >= static_cast<int>(rgb.height)) {
        return;
    }

    const uint8_t* rgb_row = rgb.pixels + y * rgb.stride;
    uint8_t* gray_row = gray.pixels + y * gray.stride;

    const int rgb_off = x * rgb.channels;
    const uint8_t r = rgb_row[rgb_off + 0];
    const uint8_t g = rgb_row[rgb_off + 1];
    const uint8_t b = rgb_row[rgb_off + 2];

    const uint8_t y8 = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    gray_row[x] = y8;
}

void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image) {
    // YOUR CODE HERE
    // NB: output gray_device_image is expected to be preallocated, no need to do any allocations
    // here NB: no explicit cudaDeviceSynchronize is required here

    // sanity: dimensions match, gray.channels == 1, rgb.channels >=3
    dim3 block(16, 16);
    dim3 grid((rgb_device_image.width + block.x - 1) / block.x,
              (rgb_device_image.height + block.y - 1) / block.y);

    GrayscaleKernel<<<grid, block>>>(rgb_device_image, gray_device_image);
}

void FreeDeviceImage(const Image& image) {
    auto ptr = image.pixels;
    cudaFree(ptr);
}

void FreeHostImage(const Image& image) {
    auto ptr = image.pixels;
    free(ptr);
}
