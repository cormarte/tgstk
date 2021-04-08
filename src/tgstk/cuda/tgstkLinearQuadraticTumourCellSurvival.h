#ifndef TGSTKLINEARQUADRATICTUMOURCELLSURVIVAL_H
#define TGSTKLINEARQUADRATICTUMOURCELLSURVIVAL_H

void gpuLinearQuadraticTumourCellSurvival(float* hostDoseMap, float* hostInitialDensity, float* hostFinalDensity, int* dimensions, float alpha, float beta);

#endif // TGSTKLINEARQUADRATICTUMOURCELLSURVIVAL_H
