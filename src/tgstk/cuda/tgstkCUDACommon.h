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

#ifndef TGSTKCUDACOMMON_H
#define TGSTKCUDACOMMON_H

#include <stdio.h>

#define CUDA_CHECK(call) {                                                                \
                                                                                     \
    const cudaError_t error = call;                                                  \
                                                                                     \
    if (error != cudaSuccess) {                                                      \
                                                                                     \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
        fprintf(stderr, "Code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    }                                                                                \
}

#endif // TGSTKCUDACOMMON_H
