#ifndef TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONSTANDARDSTENCIL_H
#define TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONSTANDARDSTENCIL_H

void gpuFiniteDifferenceReactionDiffusionStandardStencil(float* hostDxx, float* hostDxy, float* hostDxz, float* hostDyy, float* hostDyz, float* hostDzz, float* hostProliferationRate, unsigned char* hostBoundary, float* hostInitialDensity, float* hostFinalDensity, float* hostFinalDensityGradientX, float* hostFinalDensityGradientY, float* hostFinalDensityGradientZ, int* dimensions, float* spacing, int numberOfIterations, float timeStep);

#endif // TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONSTANDARDSTENCIL_H
