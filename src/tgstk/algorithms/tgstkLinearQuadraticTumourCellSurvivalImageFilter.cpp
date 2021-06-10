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

#include <tgstk/algorithms/tgstkLinearQuadraticTumourCellSurvivalImageFilter.h>
#include <tgstk/cuda/tgstkLinearQuadraticTumourCellSurvival.h>

#include <chrono>
#include <limits>

using namespace std::chrono;

tgstkLinearQuadraticTumourCellSurvivalImageFilter::tgstkLinearQuadraticTumourCellSurvivalImageFilter() {

    this->objectName = "tgstkLinearQuadraticTumourCellSurvivalImageFilter";

    this->alpha = 0.15;
    this->beta = 0.03;

    this->doseMapImage = nullptr;
    this->finalCellDensityImage = nullptr;
    this->initialCellDensityImage = nullptr;
}

tgstkLinearQuadraticTumourCellSurvivalImageFilter::~tgstkLinearQuadraticTumourCellSurvivalImageFilter() {

}

bool tgstkLinearQuadraticTumourCellSurvivalImageFilter::check() {

    // Parameters

    if (!assertValueInRange(alpha, 0.0, std::numeric_limits<double>::max())) return false;
    if (!assertValueInRange(beta, 0.0, std::numeric_limits<double>::max())) return false;


    // Dose map

    if (!assertNotNullPtr(doseMapImage)) return false;
    if (!assertImageScalarType(doseMapImage, std::vector<int>({VTK_DOUBLE}))) return false;
    if (!assertImageNumberOfScalarComponents(doseMapImage, std::vector<int>({1}))) return false;


    // Initial density

    if (!assertNotNullPtr(initialCellDensityImage)) return false;
    if (!assertImageScalarType(initialCellDensityImage, std::vector<int>({VTK_DOUBLE}))) return false;
    if (!assertImageNumberOfScalarComponents(initialCellDensityImage, std::vector<int>({1}))) return false;   // Image geometries


    // Image geometries

    if (!assertEqualImageDimensions({doseMapImage, initialCellDensityImage})) return false;
    if (!assertEqualImageSpacings({doseMapImage, initialCellDensityImage})) return false;

    return true;
}

void tgstkLinearQuadraticTumourCellSurvivalImageFilter::execute() {

    // Dimensions & spacing

    int* dimensions = this->initialCellDensityImage->GetDimensions();


    // Data cast and copy
    // TODO: Avoid copy using vtkImageData::GetScalarPointer() or vtkImageExport?

    size_t size = dimensions[0]*dimensions[1]*dimensions[2];

    float* doseMapArray = new float[size];
    float* initialCellDensityArray = new float[size];
    float* finalCellDensityArray = new float[size];

    size_t index = 0;

    for (int z=0; z<dimensions[2]; z++) {

        for (int y=0; y<dimensions[1]; y++) {

            for (int x=0; x<dimensions[0]; x++) {

                doseMapArray[index] = (float)(static_cast<double*>(this->doseMapImage->GetScalarPointer(x, y, z))[0]);
                initialCellDensityArray[index] = (float)(static_cast<double*>(this->initialCellDensityImage->GetScalarPointer(x, y, z))[0]);

                index++;
            }
        }
    }


    // GPU simulation

    high_resolution_clock::time_point tic = high_resolution_clock::now();
    gpuLinearQuadraticTumourCellSurvival(doseMapArray, initialCellDensityArray, finalCellDensityArray, dimensions, (float)(this->alpha), (float)(this->beta));
    high_resolution_clock::time_point toc = high_resolution_clock::now();
    cout << this->objectName << ": Info: Dose map applied in " << duration_cast<duration<double>>(toc - tic).count() << " seconds." << endl;


    // Data copy

    this->finalCellDensityImage = this->getNewImageFromReferenceImage(this->initialCellDensityImage, VTK_DOUBLE, 1);

    this->fillImage(this->finalCellDensityImage, 0.0);

    index = 0;

    for (int z=0; z<dimensions[2]; z++) {

        for (int y=0; y<dimensions[1]; y++) {

            for (int x=0; x<dimensions[0]; x++) {

                static_cast<double*>(this->finalCellDensityImage->GetScalarPointer(x, y, z))[0] = finalCellDensityArray[index];

                index++;
            }
        }
    }


    // Array deletion

    delete [] doseMapArray;
    delete [] initialCellDensityArray;
    delete [] finalCellDensityArray;
}

vtkSmartPointer<vtkImageData> tgstkLinearQuadraticTumourCellSurvivalImageFilter::getFinalCellDensityImage() {

    return this->finalCellDensityImage;
}

void tgstkLinearQuadraticTumourCellSurvivalImageFilter::setAlpha(double alpha) {

    this->alpha = alpha;
}

void tgstkLinearQuadraticTumourCellSurvivalImageFilter::setBeta(double beta) {

    this->beta = beta;
}

void tgstkLinearQuadraticTumourCellSurvivalImageFilter::setDoseMapImage(vtkSmartPointer<vtkImageData> image) {

    this->doseMapImage = image;
}

void tgstkLinearQuadraticTumourCellSurvivalImageFilter::setInitialCellDensityImage(vtkSmartPointer<vtkImageData> image) {

    this->initialCellDensityImage = image;
}
