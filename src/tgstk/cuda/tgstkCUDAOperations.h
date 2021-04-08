#ifndef TGSTKCUDAOPERATIONS_H
#define TGSTKCUDAOPERATIONS_H

#include <cuda_runtime.h>

__global__ void gpuAdd(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuConstMultiply(cudaPitchedPtr devSrc, float c, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuCopy(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuDivide(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuMultiply(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuNorm(cudaPitchedPtr devSrcX, cudaPitchedPtr devSrcY, cudaPitchedPtr devSrcZ, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuSet(cudaPitchedPtr devDest, float value, int w, int h, int d);
__global__ void gpuSubstract(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d);
__global__ void gpuThreshold(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, float threshold, int w, int h, int d);

template<int blockDimX, int blockDimY, int blockDimZ, int pixelsPerThread, int kernelBlockRatio>
__global__ void gpuConvolutionX(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, float* kernel, int kernelRadius, int w, int h, int d) {

    __shared__ float sharedData[blockDimX*(pixelsPerThread + 2*kernelBlockRatio)][blockDimY][blockDimZ]; // 10 MB


    // Base indices

    const int baseX = (blockIdx.x * pixelsPerThread - kernelBlockRatio)*blockDimX + threadIdx.x;
    const int baseY = blockIdx.y * blockDimY + threadIdx.y;
    const int baseZ = blockIdx.z * blockDimZ + threadIdx.z;

    const size_t pitch = devSrc.pitch;
    const size_t slicePitch = pitch * h;


    // Shared memory tile loading

    #pragma unroll
    for (int i = kernelBlockRatio; i != kernelBlockRatio + pixelsPerThread; i++) {

        sharedData[threadIdx.x + i*blockDimX][threadIdx.y][threadIdx.z] = (baseX + i*blockDimX < w && baseY < h && baseZ < d) ? *((float*)((char*)devSrc.ptr + baseZ * slicePitch + baseY * pitch) + baseX + i*blockDimX) : 0.0f;
    }


    // Shared memory left halo loading

    #pragma unroll
    for (int i = 0; i != kernelBlockRatio; i++) {

        sharedData[threadIdx.x + i*blockDimX][threadIdx.y][threadIdx.z] = (baseX + i*blockDimX >= 0 && baseY < h && baseZ < d) ? *((float*)((char*)devSrc.ptr + baseZ * slicePitch + baseY * pitch) + baseX + i*blockDimX) : 0.0f;
    }


    // Shared memory right halo loading

    #pragma unroll
    for (int i = kernelBlockRatio + pixelsPerThread; i != kernelBlockRatio + pixelsPerThread + kernelBlockRatio; i++) {

        sharedData[threadIdx.x + i*blockDimX][threadIdx.y][threadIdx.z] = (baseX + i*blockDimX < w && baseY < h && baseZ < d) ? *((float*)((char*)devSrc.ptr + baseZ * slicePitch + baseY * pitch) + baseX + i*blockDimX) : 0.0f;
    }

    __syncthreads();


    // Convolution

    #pragma unroll
    for (int i = kernelBlockRatio; i != kernelBlockRatio + pixelsPerThread; i++) {

        float sum = 0.0f;

        #pragma unroll
        for (int j = -kernelRadius; j != kernelRadius + 1; j++) {

            sum += kernel[kernelRadius + j] * sharedData[threadIdx.x + i*blockDimX + j][threadIdx.y][threadIdx.z];
        }

        if (baseX + i*blockDimX < w && baseY < h && baseZ < d) {

            *((float*)((char*)devDest.ptr + baseZ * slicePitch + baseY * pitch) + baseX + i*blockDimX) = sum;
        }
    }
}

template<int blockDimX, int blockDimY, int blockDimZ, int pixelsPerThread, int kernelBlockRatio>
__global__ void gpuConvolutionY(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, float* kernel, int kernelRadius, int w, int h, int d) {

    __shared__ float sharedData[blockDimX][blockDimY*(pixelsPerThread + 2*kernelBlockRatio)][blockDimZ]; // 10 MB


    // Base indices

    const int baseX = blockIdx.x * blockDimX + threadIdx.x;
    const int baseY = (blockIdx.y * pixelsPerThread - kernelBlockRatio)*blockDimY + threadIdx.y;
    const int baseZ = blockIdx.z * blockDimZ + threadIdx.z;

    const size_t pitch = devSrc.pitch;
    const size_t slicePitch = pitch * h;


    // Shared memory tile loading

    #pragma unroll
    for (int i = kernelBlockRatio; i != kernelBlockRatio + pixelsPerThread; i++) {

        sharedData[threadIdx.x][threadIdx.y + i*blockDimY][threadIdx.z] = (baseX < w && baseY + i*blockDimY < h && baseZ < d) ? *((float*)((char*)devSrc.ptr + baseZ * slicePitch + (baseY + i*blockDimY) * pitch) + baseX) : 0.0f;
    }


    // Shared memory left halo loading

    #pragma unroll
    for (int i = 0; i != kernelBlockRatio; i++) {

        sharedData[threadIdx.x][threadIdx.y + i*blockDimY][threadIdx.z] = (baseX < w && baseY + i*blockDimY >= 0 && baseZ < d) ? *((float*)((char*)devSrc.ptr + baseZ * slicePitch + (baseY + i*blockDimY) * pitch) + baseX) : 0.0f;
    }


    // Shared memory right halo loading

    #pragma unroll
    for (int i = kernelBlockRatio + pixelsPerThread; i != kernelBlockRatio + pixelsPerThread + kernelBlockRatio; i++) {

        sharedData[threadIdx.x][threadIdx.y + i*blockDimY][threadIdx.z] = (baseX < w && baseY + i*blockDimY < h && baseZ < d) ? *((float*)((char*)devSrc.ptr + baseZ * slicePitch + (baseY + i*blockDimY) * pitch) + baseX) : 0.0f;
    }

    __syncthreads();


    // Convolution

    #pragma unroll
    for (int i = kernelBlockRatio; i != kernelBlockRatio + pixelsPerThread; i++) {

        float sum = 0.0f;

        #pragma unroll
        for (int j = -kernelRadius; j != kernelRadius + 1; j++) {

            sum += kernel[kernelRadius + j] * sharedData[threadIdx.x][threadIdx.y + i*blockDimY + j][threadIdx.z];
        }

        if (baseX < w && baseY + i*blockDimY < h && baseZ < d) {

            *((float*)((char*)devDest.ptr + baseZ * slicePitch + (baseY + i*blockDimY) * pitch) + baseX) = sum;
        }
    }
}

template<int blockDimX, int blockDimY, int blockDimZ, int pixelsPerThread, int kernelBlockRatio>
__global__ void gpuConvolutionZ(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, float* kernel, int kernelRadius, int w, int h, int d) {

    __shared__ float sharedData[blockDimX][blockDimY][blockDimZ*(pixelsPerThread + 2*kernelBlockRatio)]; // 10 MB


    // Base indices

    const int baseX = blockIdx.x * blockDimX + threadIdx.x;
    const int baseY = blockIdx.y * blockDimY + threadIdx.y;
    const int baseZ = (blockIdx.z * pixelsPerThread - kernelBlockRatio)*blockDimZ + threadIdx.z;

    const size_t pitch = devSrc.pitch;
    const size_t slicePitch = pitch * h;


    // Shared memory tile loading

    #pragma unroll
    for (int i = kernelBlockRatio; i != kernelBlockRatio + pixelsPerThread; i++) {

        sharedData[threadIdx.x][threadIdx.y][threadIdx.z + i*blockDimZ] = (baseX < w && baseY < h && baseZ + i*blockDimZ < d) ? *((float*)((char*)devSrc.ptr + (baseZ + i*blockDimZ) * slicePitch + baseY * pitch) + baseX) : 0.0f;
    }


    // Shared memory left halo loading

    #pragma unroll
    for (int i = 0; i != kernelBlockRatio; i++) {

        sharedData[threadIdx.x][threadIdx.y][threadIdx.z + i*blockDimZ] = (baseX < w && baseY < h && baseZ + i*blockDimZ >= 0) ? *((float*)((char*)devSrc.ptr + (baseZ + i*blockDimZ) * slicePitch + baseY * pitch) + baseX) : 0.0f;
    }


    // Shared memory right halo loading

    #pragma unroll
    for (int i = kernelBlockRatio + pixelsPerThread; i != kernelBlockRatio + pixelsPerThread + kernelBlockRatio; i++) {

        sharedData[threadIdx.x][threadIdx.y][threadIdx.z + i*blockDimZ] = (baseX < w && baseY < h && baseZ + i*blockDimZ < d) ? *((float*)((char*)devSrc.ptr + (baseZ + i*blockDimZ) * slicePitch + baseY * pitch) + baseX) : 0.0f;
    }

    __syncthreads();


    // Convolution

    #pragma unroll
    for (int i = kernelBlockRatio; i != kernelBlockRatio + pixelsPerThread; i++) {

        float sum = 0.0f;

        #pragma unroll
        for (int j = -kernelRadius; j != kernelRadius + 1; j++) {

            sum += kernel[kernelRadius + j] * sharedData[threadIdx.x][threadIdx.y][threadIdx.z + i*blockDimZ + j];
        }

        if (baseX < w && baseY < h && baseZ + i*blockDimZ < d) {

            *((float*)((char*)devDest.ptr + (baseZ + i*blockDimZ) * slicePitch + baseY * pitch) + baseX) = sum;
        }
    }
}

template<int blockDimX, int blockDimY, int blockDimZ, int pixelsPerDimension>
__global__ void gpuReduce(cudaPitchedPtr devSrc, float* devDest, int w, int h, int d) {

    const int numberOfThreads = blockDimX*blockDimY*blockDimZ;
    const int sharedSize = numberOfThreads*pixelsPerDimension*pixelsPerDimension*pixelsPerDimension;

    __shared__ float sharedData[sharedSize];

    const int localThreadIndex1D = blockDimX*blockDimY*threadIdx.z + blockDimX*threadIdx.y + threadIdx.x;
    const int blockIndex1D = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;

    const int baseX = blockIdx.x * pixelsPerDimension * blockDimX + threadIdx.x;
    const int baseY = blockIdx.y * pixelsPerDimension * blockDimY + threadIdx.y;
    const int baseZ = blockIdx.z * pixelsPerDimension * blockDimZ + threadIdx.z;

    const size_t pitch = devSrc.pitch;
    const size_t slicePitch = pitch * h;


    // Copy

    #pragma unroll
    for (int k=0; k!=pixelsPerDimension; k++) {

        #pragma unroll
        for (int j=0; j!=pixelsPerDimension; j++) {

            #pragma unroll
            for (int i=0; i!=pixelsPerDimension; i++) {

                sharedData[blockDimX*pixelsPerDimension*blockDimY*pixelsPerDimension*(threadIdx.z + k*blockDimZ) + blockDimX*pixelsPerDimension*(threadIdx.y + j*blockDimY) + threadIdx.x + i*blockDimX] = (baseX + i*blockDimX < w && baseY + j*blockDimY < h && baseZ + k*blockDimZ < d) ? *((float*)((char*)devSrc.ptr + (baseZ + k*blockDimZ) * slicePitch + (baseY + j*blockDimY) * pitch) + baseX + i*blockDimX) : 0.0f;
            }
        }
    }

    __syncthreads();


    // Reduction

    int index, i;

    #pragma unroll
    for (int s=sharedSize/2; s>0; s/=2) {

        i=0;

        do {

            index = localThreadIndex1D + i*numberOfThreads;

            if (index < s) {

                sharedData[index] += sharedData[index+s];
            }

            i++;
        }

        while (i < s/numberOfThreads);
    }

    __syncthreads();

    if (localThreadIndex1D == 0) {

        devDest[blockIndex1D] = sharedData[0];
    }
}

#endif // TGSTKCUDAOPERATIONS_H
