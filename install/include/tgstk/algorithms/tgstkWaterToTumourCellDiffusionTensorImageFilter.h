#ifndef TGSTKWATERTOCELLDIFFUSIONTENSORIMAGEFILTER_H
#define TGSTKWATERTOCELLDIFFUSIONTENSORIMAGEFILTER_H

#include <tgstk/algorithms/tgstkImageProcessor.h>

class TGSTK_EXPORT tgstkWaterToTumourCellDiffusionTensorImageFilter : public virtual tgstkImageProcessor{

    public:

        enum DiffusionTensorMode {

            ANISOTROPIC,
            ISOTROPIC
        };

        tgstkWaterToTumourCellDiffusionTensorImageFilter();
        ~tgstkWaterToTumourCellDiffusionTensorImageFilter();

        bool check();
        void execute();

        void setAnisotropyFactor(double factor);
        void setBrainMapImage(vtkSmartPointer<vtkImageData> image);
        void setDiffusionTensorMode(DiffusionTensorMode mode);
        void setDiffusionTensorModeToAnisotropic();
        void setDiffusionTensorModeToIsotropic();
        void setGreyMatterDiffusionRate(double diffusionRate);
        void setWaterDiffusionTensorImage(vtkSmartPointer<vtkImageData> image);
        void setWhiteMatterDiffusionRate(double diffusionRate);

    private:

        double anisotropyFactor;
        DiffusionTensorMode diffusionTensorMode;
        double greyMatterDiffusionRate;
        double whiteMatterDiffusionRate;

        vtkSmartPointer<vtkImageData> brainMapImage;
        vtkSmartPointer<vtkImageData> tumourCellDiffusionTensorImage;
        vtkSmartPointer<vtkImageData> waterDiffusionTensorImage;
};

#endif // TGSTKWATERTOCELLDIFFUSIONTENSORIMAGEFILTER_H
