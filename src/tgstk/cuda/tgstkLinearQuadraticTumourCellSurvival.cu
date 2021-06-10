/*==========================================================================

  This file is part of the Tumor Growth Simulation ToolKit (TGSTK)
  (<https://github.com/cormarte/TGSTK>, <https://cormarte.github.io/TGSTK>).

  Copyright (C) 2021  Corentin Martens

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <https://www.gnu.org/licenses/>.

  Contact: corentin.martens@ulb.be

==========================================================================*/

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

    CUDA_CHECK(cudaSetDevice(0));


    // Memory allocation

    cudaExtent floatExtent = make_cudaExtent(w * sizeof(float), h, d);

    cudaPitchedPtr devDoseMap;
    cudaPitchedPtr devInitialDensity;
    cudaPitchedPtr devFinalDensity;

    CUDA_CHECK(cudaMalloc3D(&devDoseMap, floatExtent));
    CUDA_CHECK(cudaMalloc3D(&devInitialDensity, floatExtent));
    CUDA_CHECK(cudaMalloc3D(&devFinalDensity, floatExtent));


    // Host to device copy

    cudaMemcpy3DParms hostToDeviceParameters = {0};

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostDoseMap, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devDoseMap;
    hostToDeviceParameters.extent = floatExtent;
    CUDA_CHECK(cudaMemcpy3D(&hostToDeviceParameters));

    hostToDeviceParameters.kind = cudaMemcpyHostToDevice;
    hostToDeviceParameters.srcPtr = make_cudaPitchedPtr(hostInitialDensity, w * sizeof(float), w, h);
    hostToDeviceParameters.dstPtr = devInitialDensity;
    hostToDeviceParameters.extent = floatExtent;
    CUDA_CHECK(cudaMemcpy3D(&hostToDeviceParameters));


    // Kernel

    gpuLinearQuadraticTumourCellSurvivalKernel<<<gridDim, blockDim>>>(devDoseMap, devInitialDensity, devFinalDensity, alpha, beta, w, h, d);


    // Device to host copy

    cudaMemcpy3DParms deviceToHostParameters = {0};

    deviceToHostParameters.srcPtr = devFinalDensity;
    deviceToHostParameters.dstPtr = make_cudaPitchedPtr(hostFinalDensity, w * sizeof(float), w, h);
    deviceToHostParameters.extent = floatExtent;
    deviceToHostParameters.kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy3D(&deviceToHostParameters));


    // Memory deallocation

    CUDA_CHECK(cudaFree(devDoseMap.ptr));
    CUDA_CHECK(cudaFree(devInitialDensity.ptr));
    CUDA_CHECK(cudaFree(devFinalDensity.ptr));


    // Reset

    CUDA_CHECK(cudaDeviceReset());
}
