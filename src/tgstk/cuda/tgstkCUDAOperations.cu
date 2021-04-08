#include "tgstkCUDACommon.h"
#include <cuda_runtime.h>


//**********************************************************************************************************************//
//                                                   Const Memory Set                                                   //
//**********************************************************************************************************************//

__global__ void gpuSet(cudaPitchedPtr devDest, float value, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t pitch = devDest.pitch;
    const size_t slicePitch = pitch * h;

    if (x < w && y < h && z < d) {

        *((float*)((char*)devDest.ptr + z * slicePitch + y * pitch) + x) = value;
    }
}


//**********************************************************************************************************************//
//                                                         Copy                                                         //
//**********************************************************************************************************************//

__global__ void gpuCopy(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t pitch = devDest.pitch;
    const size_t slicePitch = pitch * h;

    if (x < w && y < h && z < d) {

        float value = *((float*)((char*)devSrc.ptr + (z * h + y) * devSrc.pitch) + x);
        *((float*)((char*)devDest.ptr + z * slicePitch + y * pitch) + x) = value;
    }
}


//**********************************************************************************************************************//
//                                                       Addition                                                       //
//**********************************************************************************************************************//

__global__ void gpuAdd(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float value1 = *((float*)((char*)devSrc1.ptr + (z * h + y) * devSrc1.pitch) + x);
        float value2 = *((float*)((char*)devSrc2.ptr + (z * h + y) * devSrc2.pitch) + x);
        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = value1+value2;
    }
}


//**********************************************************************************************************************//
//                                                     Substraction                                                     //
//**********************************************************************************************************************//

__global__ void gpuSubstract(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float value1 = *((float*)((char*)devSrc1.ptr + (z * h + y) * devSrc1.pitch) + x);
        float value2 = *((float*)((char*)devSrc2.ptr + (z * h + y) * devSrc2.pitch) + x);
        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = value1-value2;
    }
}


//**********************************************************************************************************************//
//                                                 Const Multiplication                                                 //
//**********************************************************************************************************************//

__global__ void gpuConstMultiply(cudaPitchedPtr devSrc, float c, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float value = *((float*)((char*)devSrc.ptr + (z * h + y) * devSrc.pitch) + x);
        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = value*c;
    }
}


//**********************************************************************************************************************//
//                                                    Multiplication                                                    //
//**********************************************************************************************************************//

__global__ void gpuMultiply(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float value1 = *((float*)((char*)devSrc1.ptr + (z * h + y) * devSrc1.pitch) + x);
        float value2 = *((float*)((char*)devSrc2.ptr + (z * h + y) * devSrc2.pitch) + x);
        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = value1*value2;
    }
}


//**********************************************************************************************************************//
//                                                       Division                                                       //
//**********************************************************************************************************************//

__global__ void gpuDivide(cudaPitchedPtr devSrc1, cudaPitchedPtr devSrc2, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float value1 = *((float*)((char*)devSrc1.ptr + (z * h + y) * devSrc1.pitch) + x);
        float value2 = *((float*)((char*)devSrc2.ptr + (z * h + y) * devSrc2.pitch) + x);

        if (value2 == 0.0f) {

            value2 = 1.0e-15f;
        }

        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = value1/value2;
    }
}


//**********************************************************************************************************************//
//                                                       Norm                                                           //
//**********************************************************************************************************************//

__global__ void gpuNorm(cudaPitchedPtr devSrcX, cudaPitchedPtr devSrcY, cudaPitchedPtr devSrcZ, cudaPitchedPtr devDest, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float valueX = *((float*)((char*)devSrcX.ptr + (z * h + y) * devSrcX.pitch) + x);
        float valueY = *((float*)((char*)devSrcY.ptr + (z * h + y) * devSrcY.pitch) + x);
        float valueZ = *((float*)((char*)devSrcZ.ptr + (z * h + y) * devSrcZ.pitch) + x);
        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = norm3df(valueX, valueY, valueZ);
    }
}


//**********************************************************************************************************************//
//                                                    Threshold                                                         //
//**********************************************************************************************************************//

__global__ void gpuThreshold(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, float threshold, int w, int h, int d) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < d) {

        float value = *((float*)((char*)devSrc.ptr + (z * h + y) * devSrc.pitch) + x);
        *((float*)((char*)devDest.ptr + (z * h + y) * devDest.pitch) + x) = (value <= threshold) ? 1.0f : 0.0f;
    }
}
