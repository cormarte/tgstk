#ifndef TGSTKIMAGETOMESHFILTER_H
#define TGSTKIMAGETOMESHFILTER_H

#include <tgstk/tgstkGlobal.h>
#include <tgstk/algorithms/tgstkImageProcessorBase.h>
#include <tgstk/algorithms/tgstkMeshProcessor.h>

class TGSTK_EXPORT tgstkImageToMeshFilter : public virtual tgstkImageProcessorBase, public virtual tgstkMeshProcessorBase {

    public:

        tgstkImageToMeshFilter();
        ~tgstkImageToMeshFilter();

        bool check();
        void execute();

        void exudeOff();
        void exudeOn();
        vtkSmartPointer<vtkUnstructuredGrid> getOutputMesh();
        void lloydOff();
        void lloydOn();
        void odtOff();
        void odtOn();
        void perturbOff();
        void perturbOn();
        void setInputImage(vtkSmartPointer<vtkImageData> image);
        void setMaximumCellRadiusEdgeRatio(double maximumCellRadiusEdgeRatio);
        void setMaximumCellSize(double maximumCellSize);
        void setMaximumFacetDistance(double maximumFacetDistance);
        void setMaximumFacetSize(double maximumFacetSize);
        void setMinimumFacetAngle(double minimumFacetAngle);
        void setUseExude(bool useExude);
        void setUseLloyd(bool useLloyd);
        void setUseODT(bool useODT);
        void setUsePerturb(bool usePerturb);

    private:

        double maximumCellRadiusEdgeRatio;
        double maximumCellSize;
        double maximumFacetDistance;
        double maximumFacetSize;
        double minimumFacetAngle;
        bool useExude;
        bool useLloyd;
        bool useODT;
        bool usePerturb;

        vtkSmartPointer<vtkImageData> inputImage;
        vtkSmartPointer<vtkUnstructuredGrid> outputMesh;
};

#endif // TGSTKIMAGETOMESHFILTER_H
