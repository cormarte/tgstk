#ifndef TGSTKMESHSCALARSFROMIMAGEFILTER_H
#define TGSTKMESHSCALARSFROMIMAGEFILTER_H

#include <tgstk/tgstkGlobal.h>
#include <tgstk/algorithms/tgstkImageProcessorBase.h>
#include <tgstk/algorithms/tgstkMeshProcessor.h>

class TGSTK_EXPORT tgstkMeshScalarsFromImageFilter : public virtual tgstkImageProcessorBase, public virtual tgstkMeshProcessorBase {

    public:

        enum AssignmentMode {

            CELLS_BARYCENTER_NEAREST,
            CELLS_VERTICES_MEAN,
            POINTS_NEAREST,
        };

        tgstkMeshScalarsFromImageFilter();
        ~tgstkMeshScalarsFromImageFilter();

        bool check();
        void execute();

        void setArrayName(std::string name);
        void setAssignmentMode(AssignmentMode mode);
        void setAssignmentModeToCellsBarycenterNearest();
        void setAssignmentModeToCellsVerticesMean();
        void setAssignmentModeToPointsNearest();
        void setDefaultValue(std::vector<double> value);
        void setInputImage(vtkSmartPointer<vtkImageData> image);
        void setInputMesh(vtkSmartPointer<vtkUnstructuredGrid> mesh);

    private:

        std::string arrayName;
        AssignmentMode assignmentMode;
        std::vector<double> defaultValue;

        vtkSmartPointer<vtkImageData> inputImage;
        vtkSmartPointer<vtkUnstructuredGrid> inputMesh;
};

#endif // TGSTKMESHSCALARSFROMIMAGEFILTER_H
