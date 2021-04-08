#ifndef TGSTKLINEARQUADRATICTUMOURCELLSURVIVALIMAGEFILTER_H
#define TGSTKLINEARQUADRATICTUMOURCELLSURVIVALIMAGEFILTER_H

#include <tgstk/algorithms/tgstkImageProcessor.h>

class TGSTK_EXPORT tgstkLinearQuadraticTumourCellSurvivalImageFilter : public virtual tgstkImageProcessor {

    public:

        tgstkLinearQuadraticTumourCellSurvivalImageFilter();
        ~tgstkLinearQuadraticTumourCellSurvivalImageFilter();

        bool check();
        bool execute();

        vtkSmartPointer<vtkImageData> getDoseMapImage();
        vtkSmartPointer<vtkImageData> getFinalCellDensityImage();
        vtkSmartPointer<vtkImageData> getInitialCellDensityImage();
        void setAlpha(double alpha);
        void setBeta(double beta);
        void setDoseMapImage(vtkSmartPointer<vtkImageData> image);
        void setInitialCellDensityImage(vtkSmartPointer<vtkImageData> image);

    private:

        double alpha;
        double beta;

        vtkSmartPointer<vtkImageData> doseMapImage;
        vtkSmartPointer<vtkImageData> finalCellDensityImage;
        vtkSmartPointer<vtkImageData> initialCellDensityImage;
};

#endif // TGSTKLINEARQUADRATICTUMOURCELLSURVIVALIMAGEFILTER_H
