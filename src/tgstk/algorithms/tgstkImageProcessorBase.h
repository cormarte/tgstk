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

/**
 *
 * @class tgstkImageProcessorBase
 *
 * @brief Base class for TGSTK image processors.
 *
 * tgstkImageProcessorBase is a base class for TGSTK image processing
 * algorithms.
 *
 */

#ifndef TGSTKIMAGEPROCESSORBASE_H
#define TGSTKIMAGEPROCESSORBASE_H

#define assertImageNumberOfScalarComponents(image, numberOfScalarComponents) _assertImageNumberOfScalarComponents(image, numberOfScalarComponents, #image)
#define assertImageScalarType(image, scalarTypes) _assertImageScalarType(image, scalarTypes, #image)

#include <tgstk/tgstkGlobal.h>
#include <tgstk/algorithms/tgstkAlgorithmBase.h>

#include <vector>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>

class TGSTK_EXPORT tgstkImageProcessorBase : public virtual tgstkAlgorithmBase {

    public:

        virtual ~tgstkImageProcessorBase();

    protected:

        tgstkImageProcessorBase();

        bool _assertImageNumberOfScalarComponents(vtkSmartPointer<vtkImageData> image, std::vector<int> numberOfScalarComponents, std::string name);
        bool _assertImageScalarType(vtkSmartPointer<vtkImageData> image, std::vector<int> scalarTypes, std::string name);
        bool assertEqualImageDimensions(std::vector<vtkSmartPointer<vtkImageData>> images);
        bool assertEqualImageSpacings(std::vector<vtkSmartPointer<vtkImageData>> images);
        template<typename Type> static void fillImage(vtkSmartPointer<vtkImageData> image, Type value);
        static vtkSmartPointer<vtkImageData> getNewImageFromReferenceImage(vtkSmartPointer<vtkImageData> reference, int type, int numberOfComponents=1);
};

template<typename Type>
void tgstkImageProcessorBase::fillImage(vtkSmartPointer<vtkImageData> image, Type value) {

    int* dimensions = image->GetDimensions();
    int components = image->GetNumberOfScalarComponents();

    for (int z=0; z<dimensions[2]; z++) {

        for (int y=0; y<dimensions[1]; y++) {

            for (int x=0; x<dimensions[0]; x++) {

                for (int c=0; c<components; c++) {

                    static_cast<Type*>(image->GetScalarPointer(x, y, z))[c] = value;
                }
            }
        }
    }
}

#endif // TGSTKIMAGEPROCESSORBASE_H
