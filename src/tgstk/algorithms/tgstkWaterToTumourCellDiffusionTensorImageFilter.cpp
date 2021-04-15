#include <tgstk/algorithms/tgstkWaterToTumourCellDiffusionTensorImageFilter.h>
#include <tgstk/misc/tgstkBrainTissueType.h>

#include <Eigen/Dense>
#include <limits>

using namespace Eigen;

tgstkWaterToTumourCellDiffusionTensorImageFilter::tgstkWaterToTumourCellDiffusionTensorImageFilter() {

    this->objectName = "tgstkWaterToTumourCellDiffusionTensorImageFilter";

    this->anisotropyFactor = 10.0;
    this->tensorMode = ANISOTROPIC;
    this->greyMatterMeanDiffusivity = 3.0/365.0;
    this->whiteMatterMeanDiffusivity = 30.0/365.0;

    this->brainMapImage = nullptr;
    this->tumourCellDiffusionTensorImage = nullptr;
    this->waterDiffusionTensorImage = nullptr;
}

tgstkWaterToTumourCellDiffusionTensorImageFilter::~tgstkWaterToTumourCellDiffusionTensorImageFilter() {

}

bool tgstkWaterToTumourCellDiffusionTensorImageFilter::check() {

    // Parameters

    if (!assertValueInRange(anisotropyFactor, 1.0, std::numeric_limits<double>::max())) return false;
    if (!assertValueInRange(greyMatterMeanDiffusivity, 0.0, std::numeric_limits<double>::max())) return false;
    if (!assertValueInRange(whiteMatterMeanDiffusivity, 0.0, std::numeric_limits<double>::max())) return false;


    // Brain map

    if (!assertNotNullPtr(brainMapImage)) return false;
    if (!assertImageScalarType(brainMapImage, std::vector<int>({VTK_UNSIGNED_SHORT}))) return false;
    if (!assertImageNumberOfScalarComponents(brainMapImage, std::vector<int>({1}))) return false;


    // Water diffusion tensor

    if (this->tensorMode == ANISOTROPIC) {

        if (!assertNotNullPtr(waterDiffusionTensorImage)) return false;
        if (!assertImageScalarType(waterDiffusionTensorImage, std::vector<int>({VTK_DOUBLE}))) return false;
        if (!assertImageNumberOfScalarComponents(waterDiffusionTensorImage, std::vector<int>({6, 9}))) return false;
    }

    return true;
}

void tgstkWaterToTumourCellDiffusionTensorImageFilter::execute() {

    // Dimensions

    int* dimensions = brainMapImage->GetDimensions();


    // Processing

    this->tumourCellDiffusionTensorImage = this->getNewImageFromReferenceImage(this->brainMapImage, VTK_DOUBLE, 6);
    this->fillImage(this->tumourCellDiffusionTensorImage, 0.0);

    for (int z=0; z<dimensions[2]; z++) {

        for (int y=0; y<dimensions[1]; y++) {

            for (int x=0; x<dimensions[0]; x++) {

                tgstkBrainTissueType brainTissue = static_cast<tgstkBrainTissueType*>(this->brainMapImage->GetScalarPointer(x, y, z))[0];

                double dxx, dxy, dxz, dyy, dyz, dzz;

                switch (brainTissue) {

                    case GREY_MATTER: {

                        dxx = this->greyMatterMeanDiffusivity;
                        dxy = 0.0;
                        dxz = 0.0;
                        dyy = this->greyMatterMeanDiffusivity;
                        dyz = 0.0;
                        dzz = this->greyMatterMeanDiffusivity;
                    }
                    break;

                    case WHITE_MATTER: case OEDEMA: case ENHANCING_CORE: {

                        if (this->tensorMode == ISOTROPIC) {

                            dxx = this->whiteMatterMeanDiffusivity;
                            dxy = 0.0;
                            dxz = 0.0;
                            dyy = this->whiteMatterMeanDiffusivity;
                            dyz = 0.0;
                            dzz = this->whiteMatterMeanDiffusivity;
                        }

                        else {

                            int numberOfComponents = this->waterDiffusionTensorImage->GetNumberOfScalarComponents();
                            double* voxel = static_cast<double*>(waterDiffusionTensorImage->GetScalarPointer(x, y, z));

                            Matrix<double, 3, 3> waterDiffusionTensor;
                            waterDiffusionTensor(0,0) = voxel[0];
                            waterDiffusionTensor(0,1) = voxel[1];
                            waterDiffusionTensor(0,2) = voxel[2];
                            waterDiffusionTensor(1,0) = numberOfComponents == 9 ? voxel[3] : voxel[1];
                            waterDiffusionTensor(1,1) = numberOfComponents == 9 ? voxel[4] : voxel[3];
                            waterDiffusionTensor(1,2) = numberOfComponents == 9 ? voxel[5] : voxel[4];
                            waterDiffusionTensor(2,0) = numberOfComponents == 9 ? voxel[6] : voxel[2];
                            waterDiffusionTensor(2,1) = numberOfComponents == 9 ? voxel[7] : voxel[4];
                            waterDiffusionTensor(2,2) = numberOfComponents == 9 ? voxel[8] : voxel[5];

                            SelfAdjointEigenSolver<Matrix<double, 3, 3>> solver(waterDiffusionTensor);
                            auto eigenValues = solver.eigenvalues();
                            auto eigenVectors = solver.eigenvectors();

                            // Clipping negative eigen values to zero is usually performed in commerical softwares.
                            // See Koay et al., Magnetic Resonance in Medicine 55(4):930-936.
                            double l1 = (eigenValues[2] < 0.0 || isnan(eigenValues[2])) ? 0.0 : eigenValues[2];
                            double l2 = (eigenValues[1] < 0.0 || isnan(eigenValues[1])) ? 0.0 : eigenValues[1];
                            double l3 = (eigenValues[0] < 0.0 || isnan(eigenValues[0])) ? 0.0 : eigenValues[0];

                            Vector3d v1 = eigenVectors.col(2);
                            Vector3d v2 = eigenVectors.col(1);
                            Vector3d v3 = eigenVectors.col(0);

                            double sum = l1 + l2 + l3;

                            if (sum > std::numeric_limits<double>::epsilon()) {

                                double r = this->anisotropyFactor;

                                double cl = (l1-l2)/sum;
                                double cp = 2.0*(l2-l3)/sum;
                                double cs = 3.0*l3/sum;

                                double a1l1 = (r*cl + r*cp + cs)*l1;
                                double a2l2 = (cl + r*cp + cs)*l2;
                                double a3l3 = (cl + cp + cs)*l3;

                                double factor = 3.0*this->whiteMatterMeanDiffusivity/(a1l1 + a2l2 + a3l3);

                                dxx = factor*(a1l1*v1[0]*v1[0] + a2l2*v2[0]*v2[0] + a3l3*v3[0]*v3[0]);
                                dxy = factor*(a1l1*v1[0]*v1[1] + a2l2*v2[0]*v2[1] + a3l3*v3[0]*v3[1]);
                                dxz = factor*(a1l1*v1[0]*v1[2] + a2l2*v2[0]*v2[2] + a3l3*v3[0]*v3[2]);
                                dyy = factor*(a1l1*v1[1]*v1[1] + a2l2*v2[1]*v2[1] + a3l3*v3[1]*v3[1]);
                                dyz = factor*(a1l1*v1[1]*v1[2] + a2l2*v2[1]*v2[2] + a3l3*v3[1]*v3[2]);
                                dzz = factor*(a1l1*v1[2]*v1[2] + a2l2*v2[2]*v2[2] + a3l3*v3[2]*v3[2]);
                            }

                            else {  // Eigen values sum is too small, no diffusion

                                dxx = 0.0;
                                dxy = 0.0;
                                dxz = 0.0;
                                dyy = 0.0;
                                dyz = 0.0;
                                dzz = 0.0;
                            }
                        }
                    }
                    break;

                    default: {

                        dxx = 0.0;
                        dxy = 0.0;
                        dxz = 0.0;
                        dyy = 0.0;
                        dyz = 0.0;
                        dzz = 0.0;
                    }
                    break;
                }

                static_cast<double*>(this->tumourCellDiffusionTensorImage->GetScalarPointer(x, y, z))[0] = dxx;
                static_cast<double*>(this->tumourCellDiffusionTensorImage->GetScalarPointer(x, y, z))[1] = dxy;
                static_cast<double*>(this->tumourCellDiffusionTensorImage->GetScalarPointer(x, y, z))[2] = dxz;
                static_cast<double*>(this->tumourCellDiffusionTensorImage->GetScalarPointer(x, y, z))[3] = dyy;
                static_cast<double*>(this->tumourCellDiffusionTensorImage->GetScalarPointer(x, y, z))[4] = dyz;
                static_cast<double*>(this->tumourCellDiffusionTensorImage->GetScalarPointer(x, y, z))[5] = dzz;
            }
        }
    }
}
