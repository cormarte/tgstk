#ifndef TGSTKCUDACOMMON_H
#define TGSTKCUDACOMMON_H

#include <stdio.h>

#define CHECK(call) {                                                                \
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
