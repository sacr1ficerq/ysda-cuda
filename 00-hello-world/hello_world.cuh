#pragma once

#include <cstdio>
#include <stdexcept>

#include <cuda_helpers.h>

__global__ void HelloWorldKernel() {
    printf("Hello, world!");
}

void CallHelloWorld() {
    // Call your CUDA kernel here
    HelloWorldKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
