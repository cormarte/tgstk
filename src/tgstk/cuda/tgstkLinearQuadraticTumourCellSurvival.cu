#include "tgstkCUDACommon.h"

#include <cuda_runtime.h>


__global__ void gpuLinearQuadraticTumourCellSurvivalKernel(cudaPitchedPtr devDoseMap, cudaPitchedPtr devInitialDensity, cudaPitchedPtr devFinalDensity, float alpha, float beta, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float density = *((float*)((char*)devInitialDensity.ptr + (z * h + y) * devInitialDensity.pitch) + x);
        float dose = *((float*)((char*)devDoseMap.ptr + (z * h + y) * devDoseMap.pitch) + x);
        *((float*)((char*)devFinalDensity.ptr + (z * h + y) * devFinalDensity.pitch) + x) = density*__expf(-alpha*dose-beta*dose*dose);
    }
}

void gpuLinearQuadraticTumourCellSurvival(float* hostDoseMap, float* hostInitialDensity, float* hostFinalDensity, int* dimensions, float alpha, float beta) {

    // Image dimensions

    int w = dimensions[0];
    int h = dimensions[1];
    int d = dimensions[2];


    // Blocks

    const int blockDimX = 8;
    const int blockDimY = 8;
    const int blockDimZ = 8;

    const dim3 blockDim = dim3(blockDimX, blockDimY, blockDimZ);
    const dim3 gridDim = dim3((w+blockDimX-1)/blockDimX, (h+blockDimY-1)/blockDimY, (d+blockDimZ-1)/blockDimZ);


    // Device selection

    CHECK(cudaSetDevice(0));


    // Memory allocation

    cudaExtent floatExtent = make_cudaExtent(w * sizeof(float), h, d);

    cudaPitchedPtr devDoseMap;
    cudaPitchedPtr devInitialDensity;
    cudaPitchedPtr devFinalDensity;

    CHECK(cudaMalloc3D(&devDoseMap, floatExtent));
    CHECK(cudaMalloc3D(&devInitialDensity, floatExtent));
    CHECK(cudaMalloc3D(&devFinalDensity, floatExtent));


    // Host to device copy

    cudaMemcpy3DParms hostToDeviceParameters = {0};

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDoseMap, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDoseMap;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostInitialDensity, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devInitialDensity;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));


    // Kernel

    gpuLinearQuadraticTumourCellSurvivalKernel<<<gridDim, blockDim>>>(devDoseMap, devInitialDensity, devFinalDensity, alpha, beta, w, h, d);


    // Device to host copy

    cudaMemcpy3DParms deviceToHostParameters = {0};

    deviceToHostParameters.srcPtr = devFinalDensity;
    deviceToHostParameters.dstPtr = make_cudaPitchedPtr(hostFinalDensity, w * sizeof(float), w, h);
    deviceToHostParameters.extent = floatExtent;
    deviceToHostParameters.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaMemcpy3D(&deviceToHostParameters));


    // Memory deallocation

    CHECK(cudaFree(devDoseMap.ptr));
    CHECK(cudaFree(devInitialDensity.ptr));
    CHECK(cudaFree(devFinalDensity.ptr));


    // Reset

    CHECK(cudaDeviceReset());
}
