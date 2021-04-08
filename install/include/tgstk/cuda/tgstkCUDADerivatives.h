#ifndef TGSTKCUDADERIVATIVES_H
#define TGSTKCUDADERIVATIVES_H

#include <cuda_runtime.h>

void gpuDerivativeX(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
void gpuDerivativeXX(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
void gpuDerivativeY(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
void gpuDerivativeYY(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
void gpuDerivativeZ(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
void gpuDerivativeZZ(cudaPitchedPtr devSrc, cudaPitchedPtr devDest, int w, int h, int d);
void gpuDerivatives(cudaPitchedPtr devSrc, cudaPitchedPtr devDx, cudaPitchedPtr devDy, cudaPitchedPtr devDz, cudaPitchedPtr devDxx, cudaPitchedPtr devDxy, cudaPitchedPtr devDxz, cudaPitchedPtr devDyy, cudaPitchedPtr devDyz, cudaPitchedPtr devDzz, int w, int h, int d);
void gpuDivergence(cudaPitchedPtr devSrcx, cudaPitchedPtr devSrcy, cudaPitchedPtr devSrcz, cudaPitchedPtr devTemp1, cudaPitchedPtr devTemp2, cudaPitchedPtr devTemp3, cudaPitchedPtr devDest, int w, int h, int d);
void gpuGradient(cudaPitchedPtr devSrc, cudaPitchedPtr devDx, cudaPitchedPtr devDy, cudaPitchedPtr devDz, int w, int h, int d);
void gpuInitializeDerivativeKernels(float* spacing);

#endif // TGSTKCUDADERIVATIVES_H
