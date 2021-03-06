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

#include "tgstkMeshScalarsFromImageFilter.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkImageCast.h>
#include <vtkPointData.h>
#include <vtkPoints.h>

tgstkMeshScalarsFromImageFilter::tgstkMeshScalarsFromImageFilter() {

    this->objectName = "tgstkMeshScalarsFromImageFilter";

    this->arrayName = "Scalars";
    this->assignmentMode = CELLS_BARYCENTER_NEAREST;
    this->defaultValue = std::vector<double>({0.0});

    this->inputImage = nullptr;
    this->inputMesh = nullptr;
}

tgstkMeshScalarsFromImageFilter::~tgstkMeshScalarsFromImageFilter() {

}

bool tgstkMeshScalarsFromImageFilter::check() {

    if (!assertNotNullPtr(inputMesh)) return false;
    if (!assertNotNullPtr(inputImage)) return false;
    if (!assertImageScalarType(inputImage, std::vector<int>({VTK_DOUBLE}))) return false;
    if (!assertValueIsEqual(defaultValue.size(), inputImage->GetNumberOfScalarComponents())) return false;

    return true;
}

void tgstkMeshScalarsFromImageFilter::execute() {

    // Dimensions and spacing

    int* dimensions = this->inputImage->GetDimensions();
    double* origin = this->inputImage->GetOrigin();
    double* spacing = this->inputImage->GetSpacing();


    // Casting

    vtkSmartPointer<vtkImageCast> caster = vtkSmartPointer<vtkImageCast>::New();
    caster->SetOutputScalarType(VTK_DOUBLE);
    caster->SetInputData(inputImage);
    caster->Update();


    // Array allocation

    vtkIdType numberOfComponents = this->inputImage->GetNumberOfScalarComponents();
    vtkIdType numberOfTuples;

    switch (this->assignmentMode) {

        case CELLS_BARYCENTER_NEAREST:
        case CELLS_VERTICES_MEAN:

            numberOfTuples = this->inputMesh->GetNumberOfCells();
            break;

        case POINTS_NEAREST:

            numberOfTuples = this->inputMesh->GetNumberOfPoints();
            break;
    };

    vtkDoubleArray* array = vtkDoubleArray::New();
    array->SetName(this->arrayName.c_str());
    array->SetNumberOfComponents(numberOfComponents);
    array->SetNumberOfTuples(numberOfTuples);
    array->SetNumberOfValues(numberOfTuples*numberOfComponents);


    // Array filling

    vtkPoints* points = this->inputMesh->GetPoints();

    switch (this->assignmentMode) {

        case CELLS_BARYCENTER_NEAREST: {

            vtkCellArray* cells = this->inputMesh->GetCells();
            vtkIdType cellLocation = 0;

            for (size_t i=0; i!=numberOfTuples; i++) {

                vtkIdType numIds;
                const vtkIdType* pointIds;
                cells->GetCell(cellLocation, numIds, pointIds);

                double barycenter[3];
                barycenter[0] = 0.0;
                barycenter[1] = 0.0;
                barycenter[2] = 0.0;

                for (size_t j=0; j!=numIds; j++) {

                    double point[3];
                    points->GetPoint(pointIds[j], point);

                    for (int k=0; k!=3; k++) {

                        barycenter[k] += point[k];
                    }
                }

                int voxel[3];
                bool valid;

                for (int j=0; j!=3; j++) {

                    voxel[j] = (int)((barycenter[j]/numIds - origin[j])/spacing[j] + 0.5);

                    if (voxel[j] < 0 || voxel[j] >= dimensions[j]) {

                        valid = false;
                        break;
                    }
                }

                double* values = new double[numberOfComponents];

                for (int j=0; j!=numberOfComponents; j++) {

                    values[j] = valid ? static_cast<double*>(inputImage->GetScalarPointer(voxel[0], voxel[1], voxel[2]))[j] : this->defaultValue[j];
                }

                array->SetTuple(i, values);
                delete[] values;

                cellLocation += 1+numIds;
            }

            this->inputMesh->GetCellData()->AddArray(array);
        }
        break;

        case CELLS_VERTICES_MEAN: {

            vtkCellArray* cells = this->inputMesh->GetCells();
            vtkIdType cellLocation = 0;

            for (size_t i=0; i!=numberOfTuples; i++) {

                double* values = new double[numberOfComponents];

                for (int j=0; j!=numberOfComponents; j++) {

                    values[j] = 0.0;
                }

                vtkIdType numIds;
                const vtkIdType* pointIds;
                cells->GetCell(cellLocation, numIds, pointIds);

                for (size_t j=0; j!=numIds; j++) {

                    double point[3];
                    points->GetPoint(pointIds[j], point);

                    int voxel[3];
                    bool valid;

                    for (int k=0; k!=3; k++) {

                        voxel[k] = (int)((point[k] - origin[k])/spacing[k] + 0.5);

                        if (voxel[k] < 0 || voxel[k] >= dimensions[k]) {

                            valid = false;
                            break;
                        }
                    }

                    for (int k=0; k!=numberOfComponents; k++) {

                        values[k] += valid ? static_cast<double*>(inputImage->GetScalarPointer(voxel[0], voxel[1], voxel[2]))[k] : this->defaultValue[k];
                    }
                }

                for (int j=0; j!=numberOfComponents; j++) {

                    values[j] /= numIds;
                }

                array->SetTuple(i, values);
                delete[] values;

                cellLocation += 1+numIds;
            }

            this->inputMesh->GetCellData()->AddArray(array);
        }
        break;

        case POINTS_NEAREST: {

            for (size_t i=0; i!=numberOfTuples; i++) {

                double point[3];
                points->GetPoint(i, point);

                int voxel[3];
                bool valid;

                for (int j=0; j!=3; j++) {

                    voxel[j] = (int)((point[j] - origin[j])/spacing[j] + 0.5);

                    if (voxel[j] < 0 || voxel[j] >= dimensions[j]) {

                        valid = false;
                        break;
                    }
                }

                double* values = new double[numberOfComponents];

                for (int j=0; j!=numberOfComponents; j++) {

                    values[j] = valid ? static_cast<double*>(inputImage->GetScalarPointer(voxel[0], voxel[1], voxel[2]))[j] : this->defaultValue[j];
                }

                array->SetTuple(i, values);
                delete[] values;
            }

            this->inputMesh->GetPointData()->AddArray(array);
        }
        break;
    }
}

void tgstkMeshScalarsFromImageFilter::setArrayName(std::string name) {

    this->arrayName = name;
}

void tgstkMeshScalarsFromImageFilter::setDefaultValue(std::vector<double> value) {

    this->defaultValue = value;
}

void tgstkMeshScalarsFromImageFilter::setAssignmentMode(AssignmentMode mode) {

    this->assignmentMode = mode;
}

void tgstkMeshScalarsFromImageFilter::setAssignmentModeToCellsBarycenterNearest() {

    this->assignmentMode = CELLS_BARYCENTER_NEAREST;
}

void tgstkMeshScalarsFromImageFilter::setAssignmentModeToCellsVerticesMean() {

    this->assignmentMode = CELLS_VERTICES_MEAN;
}

void tgstkMeshScalarsFromImageFilter::setAssignmentModeToPointsNearest() {

    this->assignmentMode = POINTS_NEAREST;
}
