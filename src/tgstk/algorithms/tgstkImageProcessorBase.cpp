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

#include <tgstk/algorithms/tgstkImageProcessorBase.h>

tgstkImageProcessorBase::tgstkImageProcessorBase() {

}

tgstkImageProcessorBase::~tgstkImageProcessorBase() {

}

bool tgstkImageProcessorBase::_assertImageNumberOfScalarComponents(vtkSmartPointer<vtkImageData> image, std::vector<int> numberOfComponents, std::string name) {

    int components = image->GetNumberOfScalarComponents();
    bool assert = std::find(numberOfComponents.begin(), numberOfComponents.end(), components) != numberOfComponents.end();

    if (!assert) {

        cout << this->objectName << ": Error: '" << name << "' has " << components << " components but " << numberOfComponents[0];

        for (int i=1; i<numberOfComponents.size(); i++) {

            if (i==numberOfComponents.size()-1) {

                cout << " or " << numberOfComponents[i];
            }

            else {

                cout << ", " << numberOfComponents[i];
            }
        }

        cout << " components is expected." << endl;
    }

    return assert;
}

bool tgstkImageProcessorBase::_assertImageScalarType(vtkSmartPointer<vtkImageData> image, std::vector<int> types, std::string name) {

    int type = image->GetScalarType();
    bool assert = std::find(types.begin(),types.end(), type) != types.end();

    if (!assert) {

        cout << this->objectName << ": Error: '" << name << "' is of type '" << vtkImageScalarTypeNameMacro(type) << "' but '" << vtkImageScalarTypeNameMacro(types[0]) << "'";

        for (int i=1; i<types.size(); i++) {

            if (i==types.size()-1) {

                cout << " or '" << vtkImageScalarTypeNameMacro(types[i]) << "'";
            }

            else {

                cout << ", '" << vtkImageScalarTypeNameMacro(types[i]) << "'";
            }
        }

        cout << " is expected." << endl;
    }

    return assert;
}

bool tgstkImageProcessorBase::assertEqualImageDimensions(std::vector<vtkSmartPointer<vtkImageData>> images) {

    int* dimensions = images[0]->GetDimensions();
    bool valid = true;

    for (vtkSmartPointer<vtkImageData> image : images) {

        for (int i=0; i<3; i++) {

            if (image->GetDimensions()[i] != dimensions[i]) {

                valid = false;
            }
        }
    }

    if (!valid) {

        cout << this->objectName << ": Error: Images must have the same dimensions (" << dimensions[0] << "x" << dimensions[1] << "x" << dimensions[2] << ")." << endl;
    }

    return valid;
}

bool tgstkImageProcessorBase::assertEqualImageSpacings(std::vector<vtkSmartPointer<vtkImageData>> images) {

    double* spacings = images[0]->GetSpacing();
    bool valid = true;

    for (vtkSmartPointer<vtkImageData> image : images) {

        for (int i=0; i<3; i++) {

            if (image->GetSpacing()[i] != spacings[i]) {

                valid = false;
            }
        }
    }

    if (!valid) {

        cout << this->objectName << ": Error: Images must have the same spacing (" << spacings[0] << "x" << spacings[1] << "x" << spacings[2] << ")." << endl;
    }

    return valid;
}

vtkSmartPointer<vtkImageData> tgstkImageProcessorBase::getNewImageFromReferenceImage(vtkSmartPointer<vtkImageData> reference, int type, int numberOfComponents) {

    vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
    image->SetDimensions(reference->GetDimensions());
    image->SetExtent(reference->GetExtent());
    image->SetOrigin(reference->GetOrigin());
    image->SetSpacing(reference->GetSpacing());
    image->AllocateScalars(type, numberOfComponents);

    return image;
}
