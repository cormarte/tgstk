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
