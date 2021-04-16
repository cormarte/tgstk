#include <tgstk/algorithms/tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter.h>
#include <tgstk/cuda/tgstkFiniteDifferenceReactionDiffusionStandardStencil.h>
#include <tgstk/misc/tgstkBrainTissueType.h>
#include <tgstk/misc/tgstkDefines.h>

#include <algorithm>
#include <chrono>
#include <limits>

using namespace std::chrono;

tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter() {

    this->objectName = "tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter";

    //this->coreProliferationRate = 43.8/365.0;
    //this->nonCoreProliferationRate = 43.8/365.0;
    this->simulatedTime = 120.0;
    this->timeStep = 0.05;

    this->brainMapImage = nullptr;
    this->finalCellDensityImage = nullptr;
    this->finalCellDensityGradientImage = nullptr;
    this->initialCellDensityImage = nullptr;
    this->proliferationRateImage = nullptr;
    this->diffusionTensorImage = nullptr;
}

tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::~tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter() {

}

bool tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::check() {

    // Parameters

    //if (!assertValueInRange(coreProliferationRate, 0.0, std::numeric_limits<double>::max())) return false;
    //if (!assertValueInRange(nonCoreProliferationRate, 0.0, std::numeric_limits<double>::max())) return false;
    if (!assertValueInRange(simulatedTime, 0.0, std::numeric_limits<double>::max())) return false;
    if (!assertValueInRange(timeStep, std::numeric_limits<double>::epsilon(), std::numeric_limits<double>::max())) return false;


    // Brain map

    if (!assertNotNullPtr(brainMapImage)) return false;
    if (!assertImageScalarType(brainMapImage, std::vector<int>({VTK_UNSIGNED_SHORT}))) return false;
    if (!assertImageNumberOfScalarComponents(brainMapImage, std::vector<int>({1}))) return false;


    // Initial density

    if (!assertNotNullPtr(initialCellDensityImage)) return false;
    if (!assertImageScalarType(initialCellDensityImage, std::vector<int>({VTK_DOUBLE}))) return false;
    if (!assertImageNumberOfScalarComponents(initialCellDensityImage, std::vector<int>({1}))) return false;


    // Proliferation rate

    if (!assertNotNullPtr(proliferationRateImage)) return false;
    if (!assertImageScalarType(proliferationRateImage, std::vector<int>({VTK_DOUBLE}))) return false;
    if (!assertImageNumberOfScalarComponents(proliferationRateImage, std::vector<int>({1}))) return false;


    // Tumour cell diffusion tensor

    if (!assertNotNullPtr(diffusionTensorImage)) return false;
    if (!assertImageScalarType(diffusionTensorImage, std::vector<int>({VTK_DOUBLE}))) return false;
    if (!assertImageNumberOfScalarComponents(diffusionTensorImage, std::vector<int>({6, 9}))) return false;


    // Image geometries

    if (!assertEqualImageDimensions({brainMapImage, initialCellDensityImage, proliferationRateImage, diffusionTensorImage})) return false;
    if (!assertEqualImageSpacings({brainMapImage, initialCellDensityImage, proliferationRateImage, diffusionTensorImage})) return false;

    return true;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::execute() {

    // Dimensions & spacing

    int* dimensions = this->brainMapImage->GetDimensions();
    float spacing[3] = {(float)(this->brainMapImage->GetSpacing()[0]), (float)(this->brainMapImage->GetSpacing()[1]), (float)(this->brainMapImage->GetSpacing()[2])};


    // Boundary image & bounds

    int bounds[6] = {dimensions[0]-1, 0, dimensions[1]-1, 0, dimensions[2]-1};
    vtkSmartPointer<vtkImageData> boundaryImage = this->getNewImageFromReferenceImage(brainMapImage, VTK_UNSIGNED_SHORT, 1);

    for (int z=0; z<dimensions[2]; z++) {

        for (int y=0; y<dimensions[1]; y++) {

            for (int x=0; x<dimensions[0]; x++) {

                tgstkBrainTissueType brainTissue = static_cast<tgstkBrainTissueType*>(brainMapImage->GetScalarPointer(x, y, z))[0];
                unsigned short boundaryValue = 0;

                if (brainTissue != BACKGROUND && brainTissue != NECROTIC_CORE && brainTissue != CSF) {  // Diffusive voxel

                    int coordinates[3] = {x, y, z};

                    if (x==0 || x==dimensions[0]-1 ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x-1, y, z))[0] == BACKGROUND ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x-1, y, z))[0] == NECROTIC_CORE ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x-1, y, z))[0] == CSF ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x+1, y, z))[0] == BACKGROUND ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x+1, y, z))[0] == NECROTIC_CORE ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x+1, y, z))[0] == CSF) {

                        boundaryValue = boundaryValue | BOUNDARY_X_FLAG;
                    }

                    if (y==0 || y==dimensions[1]-1 ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y-1, z))[0] == BACKGROUND ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y-1, z))[0] == NECROTIC_CORE ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y-1, z))[0] == CSF ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y+1, z))[0] == BACKGROUND ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y+1, z))[0] == NECROTIC_CORE ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y+1, z))[0] == CSF) {

                        boundaryValue = boundaryValue | BOUNDARY_Y_FLAG;
                    }

                    if (z==0 || z==dimensions[2]-1 ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y, z-1))[0] == BACKGROUND ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y, z-1))[0] == NECROTIC_CORE ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y, z-1))[0] == CSF ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y, z+1))[0] == BACKGROUND ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y, z+1))[0] == NECROTIC_CORE ||
                        static_cast<unsigned short*>(brainMapImage->GetScalarPointer(x, y, z+1))[0] == CSF) {

                        boundaryValue = boundaryValue | BOUNDARY_Z_FLAG;
                    }

                    for (int i=0; i!=3; i++) {

                        if (coordinates[i] < bounds[2*i]) {

                            bounds[2*i] = coordinates[i];
                        }

                        if (coordinates[i] > bounds[2*i+1]) {

                            bounds[2*i+1] = coordinates[i];
                        }
                    }
                }

                static_cast<unsigned short*>(boundaryImage->GetScalarPointer(x, y, z))[0] = boundaryValue;
            }
        }
    }

    for (int i=0; i!=3; i++) {  // Pad with 2 voxels on each side

        bounds[2*i] = std::max(bounds[2*i]-2, 0);
        bounds[2*i+1] = std::min(bounds[2*i+1]+2, dimensions[i]-1);
    }


    // Data cast and copy
    // TODO: Avoid copy using vtkImageData::GetScalarPointer() or vtkImageExport?

    int nx = bounds[1]-bounds[0]+1;
    int ny = bounds[3]-bounds[2]+1;
    int nz = bounds[5]-bounds[4]+1;

    int croppedDimensions[3] = {nx, ny, nz};

    size_t size = nx*ny*nz;

    float* dxxArray = new float[size];
    float* dxyArray = new float[size];
    float* dxzArray = new float[size];
    float* dyyArray = new float[size];
    float* dyzArray = new float[size];
    float* dzzArray = new float[size];
    float* proliferationRateArray = new float[size];
    unsigned char* boundaryArray = new unsigned char[size];
    float* initialCellDensityArray = new float[size];
    float* finalCellDensityArray = new float[size];
    float* finalCellDensityGradientXArray = new float[size];
    float* finalCellDensityGradientYArray = new float[size];
    float* finalCellDensityGradientZArray = new float[size];

    size_t index = 0;

    for (int z=bounds[4]; z!=bounds[5]+1; z++) {

        for (int y=bounds[2]; y!=bounds[3]+1; y++) {

            for (int x=bounds[0]; x!=bounds[1]+1; x++) {

                dxxArray[index] = (float)(static_cast<double*>(this->diffusionTensorImage->GetScalarPointer(x, y, z))[0]);
                dxyArray[index] = (float)(static_cast<double*>(this->diffusionTensorImage->GetScalarPointer(x, y, z))[1]);
                dxzArray[index] = (float)(static_cast<double*>(this->diffusionTensorImage->GetScalarPointer(x, y, z))[2]);
                dyyArray[index] = (float)(static_cast<double*>(this->diffusionTensorImage->GetScalarPointer(x, y, z))[3]);
                dyzArray[index] = (float)(static_cast<double*>(this->diffusionTensorImage->GetScalarPointer(x, y, z))[4]);
                dzzArray[index] = (float)(static_cast<double*>(this->diffusionTensorImage->GetScalarPointer(x, y, z))[5]);
                proliferationRateArray[index] = (float)(static_cast<double*>(this->proliferationRateImage->GetScalarPointer(x, y, z))[0]);
                boundaryArray[index] = (unsigned char)(static_cast<unsigned short*>(boundaryImage->GetScalarPointer(x, y, z))[0]);
                initialCellDensityArray[index] = (float)(static_cast<double*>(this->initialCellDensityImage->GetScalarPointer(x, y, z))[0]);

                index++;
            }
        }
    }


    // GPU simulation

    high_resolution_clock::time_point tic = high_resolution_clock::now();
    gpuFiniteDifferenceReactionDiffusionStandardStencil(dxxArray, dxyArray, dxzArray, dyyArray, dyzArray, dzzArray, proliferationRateArray, boundaryArray, initialCellDensityArray, finalCellDensityArray, finalCellDensityGradientXArray, finalCellDensityGradientYArray, finalCellDensityGradientZArray, croppedDimensions, spacing, (int)(this->simulatedTime/this->timeStep + 0.5), (float)(this->timeStep));
    high_resolution_clock::time_point toc = high_resolution_clock::now();
    cout << this->objectName << ": Info: Simulation performed in " << duration_cast<duration<double>>(toc - tic).count() << " seconds (" << duration_cast<duration<double>>(toc - tic).count()/(int)(this->simulatedTime/this->timeStep + 0.5) << " seconds per iteration)." << endl;


    // Data copy

    this->finalCellDensityImage = this->getNewImageFromReferenceImage(this->initialCellDensityImage, VTK_DOUBLE, 1);
    this->finalCellDensityGradientImage = this->getNewImageFromReferenceImage(this->initialCellDensityImage, VTK_DOUBLE, 3);

    this->fillImage(this->finalCellDensityImage, 0.0);
    this->fillImage(this->finalCellDensityGradientImage, 0.0);

    index = 0;

    for (int z=bounds[4]; z<bounds[5]+1; z++) {

        for (int y=bounds[2]; y<bounds[3]+1; y++) {

            for (int x=bounds[0]; x<bounds[1]+1; x++) {

                static_cast<double*>(this->finalCellDensityImage->GetScalarPointer(x, y, z))[0] = finalCellDensityArray[index];
                static_cast<double*>(this->finalCellDensityGradientImage->GetScalarPointer(x, y, z))[0] = finalCellDensityGradientXArray[index];
                static_cast<double*>(this->finalCellDensityGradientImage->GetScalarPointer(x, y, z))[1] = finalCellDensityGradientYArray[index];
                static_cast<double*>(this->finalCellDensityGradientImage->GetScalarPointer(x, y, z))[2] = finalCellDensityGradientZArray[index];

                index++;
            }
        }
    }


    // Array deletion

    delete [] dxxArray;
    delete [] dxyArray;
    delete [] dxzArray;
    delete [] dyyArray;
    delete [] dyzArray;
    delete [] dzzArray;
    delete [] proliferationRateArray;
    delete [] boundaryArray;
    delete [] initialCellDensityArray;
    delete [] finalCellDensityArray;
    delete [] finalCellDensityGradientXArray;
    delete [] finalCellDensityGradientYArray;
    delete [] finalCellDensityGradientZArray;
}

vtkSmartPointer<vtkImageData> tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::getFinalCellDensityImage() {

    return this->finalCellDensityImage;
}

vtkSmartPointer<vtkImageData> tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::getFinalCellDensityGradientImage() {

    return this->finalCellDensityGradientImage;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::setBrainMapImage(vtkImageData *image) {

    this->brainMapImage = image;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::setInitialCellDensityImage(vtkSmartPointer<vtkImageData> image) {

    this->initialCellDensityImage = image;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::setProliferationRateImage(vtkSmartPointer<vtkImageData> image) {

    this->proliferationRateImage = image;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::setSimulatedTime(double time) {

    this->simulatedTime = time;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::setTimeStep(double step) {

    this->timeStep = step;
}

void tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter::setDiffusionTensorImage(vtkSmartPointer<vtkImageData> image) {

    this->diffusionTensorImage = image;
}
