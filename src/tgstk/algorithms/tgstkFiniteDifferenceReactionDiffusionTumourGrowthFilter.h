#ifndef TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONTUMOURGROWTHFILTER_H
#define TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONTUMOURGROWTHFILTER_H

#include <tgstk/algorithms/tgstkImageProcessor.h>

class TGSTK_EXPORT tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter : public virtual tgstkImageProcessor {

    public:

        tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter();
        ~tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter();

        bool check();
        void execute();

        vtkSmartPointer<vtkImageData> getFinalCellDensityImage();
        vtkSmartPointer<vtkImageData> getFinalCellDensityGradientImage();
        vtkSmartPointer<vtkImageData> getInitialCellDensityImage();
        vtkSmartPointer<vtkImageData> getProliferationRateImage();
        vtkSmartPointer<vtkImageData> getTumourCellDiffusionTensorImage();
        void setBrainMapImage(vtkSmartPointer<vtkImageData> image);
        void setCoreProliferationRate(double proliferationRate);
        void setInitialCellDensityImage(vtkSmartPointer<vtkImageData> image);
        void setNonCoreProliferationRate(double proliferationRate);
        void setProliferationRateImage(vtkSmartPointer<vtkImageData> image);
        void setSimulatedTime(double time);
        void setTimeStep(double step);
        void setTumourDiffusionTensorImage(vtkSmartPointer<vtkImageData> image);

    private:

        double coreProliferationRate;
        double nonCoreProliferationRate;
        double simulatedTime;
        double timeStep;

        vtkSmartPointer<vtkImageData> brainMapImage;
        vtkSmartPointer<vtkImageData> finalCellDensityImage;
        vtkSmartPointer<vtkImageData> finalCellDensityGradientImage;
        vtkSmartPointer<vtkImageData> initialCellDensityImage;
        vtkSmartPointer<vtkImageData> proliferationRateImage;
        vtkSmartPointer<vtkImageData> tumourCellDiffusionTensorImage;
};

#endif // TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONTUMOURGROWTHFILTER_H
