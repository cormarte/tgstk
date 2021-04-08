#include "tgstkCUDACommon.h"
#include "tgstkCUDADerivatives.h"
#include "tgstkCUDAOperations.h"


// Definitions

const unsigned int DEFAULT_BLOCK_DIM = 8;
const unsigned int DERIVATIVE_BLOCK_DIM = 16;
const unsigned int NON_DERIVATIVE_BLOCK_DIM = 4;
const unsigned int KERNEL_RADIUS = 1;
const unsigned int KERNEL_BLOCK_RATIO = 1+KERNEL_RADIUS/DERIVATIVE_BLOCK_DIM;
const unsigned int PIXELS_PER_THREAD = 8;


// Derivative kernels

__constant__ float devDxKernel[3];
__constant__ float devDyKernel[3];
__constant__ float devDzKernel[3];
__constant__ float devDxxKernel[3];
__constant__ float devDyyKernel[3];
__constant__ float devDzzKernel[3];

float* devDxKernelPtr;
float* devDyKernelPtr;
float* devDzKernelPtr;
float* devDxxKernelPtr;
float* devDyyKernelPtr;
float* devDzzKernelPtr;


inline dim3 computeGridDim(dim3 blockDim, dim3 pixelsPerThread, dim3 size) {

    return dim3((size.x+(blockDim.x-1)*pixelsPerThread.x)/(blockDim.x*pixelsPerThread.x), (size.y+(blockDim.y-1)*pixelsPerThread.y)/(blockDim.y*pixelsPerThread.y), (size.z+(blockDim.z-1)*pixelsPerThread.z)/(blockDim.z*pixelsPerThread.z));
}

void gpuDerivativeX(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(PIXELS_PER_THREAD, 1, 1), dim3(w, h, d));

    gpuConvolutionX<DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, PIXELS_PER_THREAD, KERNEL_BLOCK_RATIO><<<gridDim, blockDim>>>(devSrc, devDest, devDxKernelPtr, KERNEL_RADIUS, w, h, d);
}

void gpuDerivativeXX(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(PIXELS_PER_THREAD, 1, 1), dim3(w, h, d));

    gpuConvolutionX<DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, PIXELS_PER_THREAD, KERNEL_BLOCK_RATIO><<<gridDim, blockDim>>>(devSrc, devDest, devDxxKernelPtr, KERNEL_RADIUS, w, h, d);
}

void gpuDerivativeY(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(1, PIXELS_PER_THREAD, 1), dim3(w, h, d));

    gpuConvolutionY<NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, PIXELS_PER_THREAD, KERNEL_BLOCK_RATIO><<<gridDim, blockDim>>>(devSrc, devDest, devDyKernelPtr, KERNEL_RADIUS, w, h, d);
}

void gpuDerivativeYY(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(1, PIXELS_PER_THREAD, 1), dim3(w, h, d));

    gpuConvolutionY<NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, PIXELS_PER_THREAD, KERNEL_BLOCK_RATIO><<<gridDim, blockDim>>>(devSrc, devDest, devDyyKernelPtr, KERNEL_RADIUS, w, h, d);
}

void gpuDerivativeZ(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(1, 1, PIXELS_PER_THREAD), dim3(w, h, d));

    gpuConvolutionZ<NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM, PIXELS_PER_THREAD, KERNEL_BLOCK_RATIO><<<gridDim, blockDim>>>(devSrc, devDest, devDzKernelPtr, KERNEL_RADIUS, w, h, d);
}

void gpuDerivativeZZ(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(1, 1, PIXELS_PER_THREAD), dim3(w, h, d));

    gpuConvolutionZ<NON_DERIVATIVE_BLOCK_DIM, NON_DERIVATIVE_BLOCK_DIM, DERIVATIVE_BLOCK_DIM, PIXELS_PER_THREAD, KERNEL_BLOCK_RATIO><<<gridDim, blockDim>>>(devSrc, devDest, devDzzKernelPtr, KERNEL_RADIUS, w, h, d);
}

void gpuDerivatives(cudaPitchedPtr devSrc, cudaPitchedPtr devDx, cudaPitchedPtr devDy, cudaPitchedPtr devDz, cudaPitchedPtr devDxx, cudaPitchedPtr devDxy, cudaPitchedPtr devDxz, cudaPitchedPtr devDyy, cudaPitchedPtr devDyz, cudaPitchedPtr devDzz, int w, int h, int d) {

    gpuDerivativeX(devSrc, devDx, w, h, d);
    gpuDerivativeY(devSrc, devDy, w, h, d);
    gpuDerivativeZ(devSrc, devDz, w, h, d);
    gpuDerivativeY(devDx, devDxy, w, h, d);
    gpuDerivativeZ(devDx, devDxz, w, h, d);
    gpuDerivativeZ(devDy, devDyz, w, h, d);
    gpuDerivativeXX(devSrc, devDxx, w, h, d);
    gpuDerivativeYY(devSrc, devDyy, w, h, d);
    gpuDerivativeZZ(devSrc, devDzz, w, h, d);
}

void gpuDivergence(cudaPitchedPtr devSrcx, cudaPitchedPtr devSrcy, cudaPitchedPtr devSrcz, cudaPitchedPtr devTemp1, cudaPitchedPtr devTemp2, cudaPitchedPtr devTemp3, cudaPitchedPtr devDest, int w, int h, int d) {

    dim3 blockDim = dim3(DEFAULT_BLOCK_DIM, DEFAULT_BLOCK_DIM, DEFAULT_BLOCK_DIM);
    dim3 gridDim = computeGridDim(blockDim, dim3(1, 1, 1), dim3(w, h, d));

    gpuDerivativeX(devSrcx, devTemp1, w, h, d);
    gpuDerivativeY(devSrcy, devTemp2, w, h, d);

    gpuAdd<<<gridDim, blockDim>>>(devTemp1, devTemp2, devTemp3, w, h, d);

    gpuDerivativeZ(devSrcz, devTemp1, w, h, d);

    gpuAdd<<<gridDim, blockDim>>>(devTemp1, devTemp3, devDest, w, h, d);
}

void gpuGradient(cudaPitchedPtr devSrc, cudaPitchedPtr devDx, cudaPitchedPtr devDy, cudaPitchedPtr devDz, int w, int h, int d) {

    gpuDerivativeX(devSrc, devDx, w, h, d);
    gpuDerivativeY(devSrc, devDy, w, h, d);
    gpuDerivativeZ(devSrc, devDz, w, h, d);
}

void gpuInitializeDerivativeKernels(float* spacing) {

    float hostDxKernel[3] = {-1.0f/(2.0f*spacing[0]), 0, 1.0f/(2.0f*spacing[0])};
    float hostDyKernel[3] = {-1.0f/(2.0f*spacing[1]), 0, 1.0f/(2.0f*spacing[1])};
    float hostDzKernel[3] = {-1.0f/(2.0f*spacing[2]), 0, 1.0f/(2.0f*spacing[2])};
    float hostDxxKernel[3] = {1.0f/(spacing[0]*spacing[0]), -2.0f/(spacing[0]*spacing[0]), 1.0f/(spacing[0]*spacing[0])};  // Better approximation than applying twice Dx, which implies x+2 and x-2 coefficients
    float hostDyyKernel[3] = {1.0f/(spacing[1]*spacing[1]), -2.0f/(spacing[1]*spacing[1]), 1.0f/(spacing[1]*spacing[1])};  // Better approximation than applying twice Dy, which implies y+2 and y-2 coefficients
    float hostDzzKernel[3] = {1.0f/(spacing[2]*spacing[2]), -2.0f/(spacing[2]*spacing[2]), 1.0f/(spacing[2]*spacing[2])};  // Better approximation than applying twice Dz, which implies z+2 and z-2 coefficients

    CHECK(cudaMallocHost((void**)&devDxKernel, 3*sizeof(float), cudaHostAllocWriteCombined));
    CHECK(cudaMallocHost((void**)&devDyKernel, 3*sizeof(float), cudaHostAllocWriteCombined));
    CHECK(cudaMallocHost((void**)&devDzKernel, 3*sizeof(float), cudaHostAllocWriteCombined));
    CHECK(cudaMallocHost((void**)&devDxxKernel, 3*sizeof(float), cudaHostAllocWriteCombined));
    CHECK(cudaMallocHost((void**)&devDyyKernel, 3*sizeof(float), cudaHostAllocWriteCombined));
    CHECK(cudaMallocHost((void**)&devDzzKernel, 3*sizeof(float), cudaHostAllocWriteCombined));

    CHECK(cudaMemcpyToSymbol(devDxKernel, hostDxKernel, 3*sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(devDyKernel, hostDyKernel, 3*sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(devDzKernel, hostDzKernel, 3*sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(devDxxKernel, hostDxxKernel, 3*sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(devDyyKernel, hostDyyKernel, 3*sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(devDzzKernel, hostDzzKernel, 3*sizeof(float), 0, cudaMemcpyHostToDevice));

    CHECK(cudaGetSymbolAddress((void**)&devDxKernelPtr, devDxKernel));
    CHECK(cudaGetSymbolAddress((void**)&devDyKernelPtr, devDyKernel));
    CHECK(cudaGetSymbolAddress((void**)&devDzKernelPtr, devDzKernel));
    CHECK(cudaGetSymbolAddress((void**)&devDxxKernelPtr, devDxxKernel));
    CHECK(cudaGetSymbolAddress((void**)&devDyyKernelPtr, devDyyKernel));
    CHECK(cudaGetSymbolAddress((void**)&devDzzKernelPtr, devDzzKernel));
}
