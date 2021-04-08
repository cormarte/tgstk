#include "tgstkCUDACommon.h"
#include "tgstkCUDADerivatives.h"
#include "tgstkCUDAOperations.h"

#include <cuda_runtime.h>

__global__ void gpuFiniteDifferenceReactionDiffusionStandardStencilKernel(cudaPitchedPtr devPreviousDensity, cudaPitchedPtr devDxx, cudaPitchedPtr devDxy, cudaPitchedPtr devDxz, cudaPitchedPtr devDyy, cudaPitchedPtr devDyz, cudaPitchedPtr devDzz,  cudaPitchedPtr devProliferationRate, cudaPitchedPtr devNextDensity, float dt, int w, int h, int d, float sx, float sy, float sz) {

    // From Mosayebi et al., 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Workshops, pp. 125-132.

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && x < w-1 && y > 0 && y < h-1 && z > 0 && z < d-1) {

        float a_ccc = *((float*)((char*)devDxx.ptr + (z * h + y) * devDxx.pitch) + x);
        float a_lcc = *((float*)((char*)devDxx.ptr + (z * h + y) * devDxx.pitch) + x-1);
        float a_rcc = *((float*)((char*)devDxx.ptr + (z * h + y) * devDxx.pitch) + x+1);

        float b_lcc = *((float*)((char*)devDxy.ptr + (z * h + y) * devDxy.pitch) + x-1);
        float b_rcc = *((float*)((char*)devDxy.ptr + (z * h + y) * devDxy.pitch) + x+1);
        float b_cdc = *((float*)((char*)devDxy.ptr + (z * h + y-1) * devDxy.pitch) + x);
        float b_cuc = *((float*)((char*)devDxy.ptr + (z * h + y+1) * devDxy.pitch) + x);

        float c_lcc = *((float*)((char*)devDxz.ptr + (z * h + y) * devDxz.pitch) + x-1);
        float c_rcc = *((float*)((char*)devDxz.ptr + (z * h + y) * devDxz.pitch) + x+1);
        float c_ccb = *((float*)((char*)devDxz.ptr + ((z-1) * h + y) * devDxz.pitch) + x);
        float c_cca = *((float*)((char*)devDxz.ptr + ((z+1) * h + y) * devDxz.pitch) + x);

        float d_ccc = *((float*)((char*)devDyy.ptr + (z * h + y) * devDyy.pitch) + x);
        float d_cdc = *((float*)((char*)devDyy.ptr + (z * h + y-1) * devDyy.pitch) + x);
        float d_cuc = *((float*)((char*)devDyy.ptr + (z * h + y+1) * devDyy.pitch) + x);

        float e_cdc = *((float*)((char*)devDyz.ptr + (z * h + y-1) * devDyz.pitch) + x);
        float e_cuc = *((float*)((char*)devDyz.ptr + (z * h + y+1) * devDyz.pitch) + x);
        float e_ccb = *((float*)((char*)devDyz.ptr + ((z-1) * h + y) * devDyz.pitch) + x);
        float e_cca = *((float*)((char*)devDyz.ptr + ((z+1) * h + y) * devDyz.pitch) + x);

        float f_ccc = *((float*)((char*)devDzz.ptr + (z * h + y) * devDzz.pitch) + x);
        float f_ccb = *((float*)((char*)devDzz.ptr + ((z-1) * h + y) * devDzz.pitch) + x);
        float f_cca = *((float*)((char*)devDzz.ptr + ((z+1) * h + y) * devDzz.pitch) + x);

        float u_ccc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y) * devPreviousDensity.pitch) + x);
        float u_lcc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y) * devPreviousDensity.pitch) + x-1);
        float u_rcc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y) * devPreviousDensity.pitch) + x+1);
        float u_cdc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y-1) * devPreviousDensity.pitch) + x);
        float u_cuc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y+1) * devPreviousDensity.pitch) + x);
        float u_ldc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y-1) * devPreviousDensity.pitch) + x-1);
        float u_luc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y+1) * devPreviousDensity.pitch) + x-1);
        float u_rdc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y-1) * devPreviousDensity.pitch) + x+1);
        float u_ruc = *((float*)((char*)devPreviousDensity.ptr + (z * h + y+1) * devPreviousDensity.pitch) + x+1);
        float u_ccb = *((float*)((char*)devPreviousDensity.ptr + ((z-1) * h + y) * devPreviousDensity.pitch) + x);
        float u_lcb = *((float*)((char*)devPreviousDensity.ptr + ((z-1) * h + y) * devPreviousDensity.pitch) + x-1);
        float u_rcb = *((float*)((char*)devPreviousDensity.ptr + ((z-1) * h + y) * devPreviousDensity.pitch) + x+1);
        float u_cdb = *((float*)((char*)devPreviousDensity.ptr + ((z-1) * h + y-1) * devPreviousDensity.pitch) + x);
        float u_cub = *((float*)((char*)devPreviousDensity.ptr + ((z-1) * h + y+1) * devPreviousDensity.pitch) + x);
        float u_cca = *((float*)((char*)devPreviousDensity.ptr + ((z+1) * h + y) * devPreviousDensity.pitch) + x);
        float u_lca = *((float*)((char*)devPreviousDensity.ptr + ((z+1) * h + y) * devPreviousDensity.pitch) + x-1);
        float u_rca = *((float*)((char*)devPreviousDensity.ptr + ((z+1) * h + y) * devPreviousDensity.pitch) + x+1);
        float u_cda = *((float*)((char*)devPreviousDensity.ptr + ((z+1) * h + y-1) * devPreviousDensity.pitch) + x);
        float u_cua = *((float*)((char*)devPreviousDensity.ptr + ((z+1) * h + y+1) * devPreviousDensity.pitch) + x);

        float h_11 = 1.0f/(2.0f*sx*sx);
        float h_22 = 1.0f/(2.0f*sy*sy);
        float h_33 = 1.0f/(2.0f*sz*sz);
        float h_12 = 1.0f/(4.0f*sx*sy);
        float h_13 = 1.0f/(4.0f*sx*sz);
        float h_23 = 1.0f/(4.0f*sy*sz);

        float div = (-e_cuc-e_ccb)*h_23*u_cub
                  + ( c_lcc+c_ccb)*h_13*u_lcb
                  + ( f_ccb+f_ccc)*h_33*u_ccb
                  + (-c_rcc-c_ccb)*h_13*u_rcb
                  + ( e_cdc+e_ccb)*h_23*u_cdb

                  + (-b_lcc-b_cuc)*h_12*u_luc
                  + ( d_cuc+d_ccc)*h_22*u_cuc
                  + ( b_rcc+b_cuc)*h_12*u_ruc
                  + ( a_lcc+a_ccc)*h_11*u_lcc
                  + (-(a_lcc+2.0f*a_ccc+a_rcc)*h_11
                     -(d_cdc+2.0f*d_ccc+d_cuc)*h_22
                     -(f_ccb+2.0f*f_ccc+f_cca)*h_33)*u_ccc
                  + ( a_rcc+a_ccc)*h_11*u_rcc
                  + ( b_lcc+b_cdc)*h_12*u_ldc
                  + ( d_cdc+d_ccc)*h_22*u_cdc
                  + (-b_rcc-b_cdc)*h_12*u_rdc

                  + ( e_cuc+e_cca)*h_23*u_cua
                  + (-c_lcc-c_cca)*h_13*u_lca
                  + ( f_cca+f_ccc)*h_33*u_cca
                  + ( c_rcc+c_cca)*h_13*u_rca
                  + (-e_cdc-e_cca)*h_23*u_cda;

        float proliferationRate = *((float*)((char*)devProliferationRate.ptr + (z * h + y) * devProliferationRate.pitch) + x);
        float previousDensity = u_ccc;
        float nextDensity = __saturatef(previousDensity + dt*(div + proliferationRate*previousDensity*(1.0f-previousDensity)));
        *((float*)((char*)devNextDensity.ptr + (z * h + y) * devNextDensity.pitch) + x) = nextDensity;
    }
}


void gpuFiniteDifferenceReactionDiffusionStandardStencil(float* hostDxx, float* hostDxy, float* hostDxz, float* hostDyy, float* hostDyz, float* hostDzz, float* hostProliferationRate, unsigned char* hostBoundary, float* hostInitialDensity, float* hostFinalDensity, float* hostFinalDensityGradientX, float* hostFinalDensityGradientY, float* hostFinalDensityGradientZ, int* dimensions, float* spacing, int numberOfIterations, float timeStep) {

    // Image dimensions

    int w = dimensions[0];
    int h = dimensions[1];
    int d = dimensions[2];


    // Image spacing

    float sx = spacing[0];
    float sy = spacing[1];
    float sz = spacing[2];


    // Blocks

    const int blockDimX = 8;
    const int blockDimY = 8;
    const int blockDimZ = 8;

    const dim3 blockDim = dim3(blockDimX, blockDimY, blockDimZ);
    const dim3 gridDim = dim3((w+blockDimX-1)/blockDimX, (h+blockDimY-1)/blockDimY, (d+blockDimZ-1)/blockDimZ);


    // Device selection

    CHECK(cudaSetDevice(0));


    // Memory allocation

    cudaExtent charExtent = make_cudaExtent(w * sizeof(unsigned char), h, d);
    cudaExtent floatExtent = make_cudaExtent(w * sizeof(float), h, d);

    cudaPitchedPtr devDxx;
    cudaPitchedPtr devDxy;
    cudaPitchedPtr devDxz;
    cudaPitchedPtr devDyy;
    cudaPitchedPtr devDyz;
    cudaPitchedPtr devDzz;
    cudaPitchedPtr devProliferationRate;
    cudaPitchedPtr devBoundary;
    cudaPitchedPtr devPreviousDensity;
    cudaPitchedPtr devNextDensity;
    cudaPitchedPtr devDx;
    cudaPitchedPtr devDy;
    cudaPitchedPtr devDz;

    CHECK(cudaMalloc3D(&devDxx, floatExtent));
    CHECK(cudaMalloc3D(&devDxy, floatExtent));
    CHECK(cudaMalloc3D(&devDxz, floatExtent));
    CHECK(cudaMalloc3D(&devDyy, floatExtent));
    CHECK(cudaMalloc3D(&devDyz, floatExtent));
    CHECK(cudaMalloc3D(&devDzz, floatExtent));
    CHECK(cudaMalloc3D(&devProliferationRate, floatExtent));
    CHECK(cudaMalloc3D(&devBoundary, charExtent));
    CHECK(cudaMalloc3D(&devPreviousDensity, floatExtent));
    CHECK(cudaMalloc3D(&devNextDensity, floatExtent));
    CHECK(cudaMalloc3D(&devDx, floatExtent));
    CHECK(cudaMalloc3D(&devDy, floatExtent));
    CHECK(cudaMalloc3D(&devDz, floatExtent));


    // Host to device copy

    cudaMemcpy3DParms hostToDeviceParameters = {0};

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDxx, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDxx;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDxy, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDxy;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDxz, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDxz;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDyy, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDyy;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDyz, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDyz;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDzz, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDzz;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostProliferationRate, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devProliferationRate;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostBoundary, w * sizeof(unsigned char), w, h);
    hostToDeviceParameters.dstPtr = devBoundary;
    hostToDeviceParameters.extent = charExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostInitialDensity, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devPreviousDensity;
    hostToDeviceParameters.extent = floatExtent;
    CHECK(cudaMemcpy3D(&hostToDeviceParameters));


    // Derivative kernel initilization

    gpuInitializeDerivativeKernels(spacing);


    // Simulation

    for (int iteration=0; iteration!=numberOfIterations; iteration++) {

        gpuFiniteDifferenceReactionDiffusionStandardStencilKernel<<<gridDim, blockDim>>>(devPreviousDensity, devDxx, devDxy, devDxz, devDyy, devDyz, devDzz, devProliferationRate, devNextDensity, timeStep, w, h, d, sx, sy, sz);
        gpuCopy<<<gridDim, blockDim>>>(devNextDensity, devPreviousDensity, w, h, d);
    }

    gpuGradient(devNextDensity, devDx, devDy, devDz, w, h, d);


    // Device to host copy

    cudaMemcpy3DParms deviceToHostParameters = {0};

    deviceToHostParameters.srcPtr = devNextDensity;
    deviceToHostParameters.dstPtr = make_cudaPitchedPtr(hostFinalDensity, w * sizeof(float), w, h);
    deviceToHostParameters.extent = floatExtent;
    deviceToHostParameters.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaMemcpy3D(&deviceToHostParameters));

    deviceToHostParameters.srcPtr = devDx;
    deviceToHostParameters.dstPtr = make_cudaPitchedPtr(hostFinalDensityGradientX, w * sizeof(float), w, h);
    deviceToHostParameters.extent = floatExtent;
    deviceToHostParameters.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaMemcpy3D(&deviceToHostParameters));

    deviceToHostParameters.srcPtr = devDy;
    deviceToHostParameters.dstPtr = make_cudaPitchedPtr(hostFinalDensityGradientY, w * sizeof(float), w, h);
    deviceToHostParameters.extent = floatExtent;
    deviceToHostParameters.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaMemcpy3D(&deviceToHostParameters));

    deviceToHostParameters.srcPtr = devDz;
    deviceToHostParameters.dstPtr = make_cudaPitchedPtr(hostFinalDensityGradientZ, w * sizeof(float), w, h);
    deviceToHostParameters.extent = floatExtent;
    deviceToHostParameters.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaMemcpy3D(&deviceToHostParameters));


    // Memory deallocation

    CHECK(cudaFree(devDxx.ptr));
    CHECK(cudaFree(devDxy.ptr));
    CHECK(cudaFree(devDxz.ptr));
    CHECK(cudaFree(devDyy.ptr));
    CHECK(cudaFree(devDyz.ptr));
    CHECK(cudaFree(devDzz.ptr));
    CHECK(cudaFree(devProliferationRate.ptr));
    CHECK(cudaFree(devBoundary.ptr));
    CHECK(cudaFree(devPreviousDensity.ptr));
    CHECK(cudaFree(devNextDensity.ptr));
    CHECK(cudaFree(devDx.ptr));
    CHECK(cudaFree(devDy.ptr));
    CHECK(cudaFree(devDz.ptr));


    // Reset

    CHECK(cudaDeviceReset());
}
